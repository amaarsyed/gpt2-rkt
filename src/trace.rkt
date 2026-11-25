#lang racket/base

;; Opt-in execution trace. The observer parameter defaults to #f.
;; Hot paths check that and skip every allocation, copy, and summary.
;; Scheduling records are the matmul policy decision. Timings are separate
;; wall-clock spans from current-inexact-monotonic-milliseconds. Nothing
;; here invents a per-future timeline.

(require json
         racket/flonum
         racket/list
         racket/string
         "source-map.rkt"
         "tensor.rkt")

(provide current-trace-observer
         current-trace-stage
         current-trace-layer
         current-trace-phase
         current-trace-capture
         make-execution-trace
         execution-trace?
         execution-trace-prefill-ms
         execution-trace-decode-ms
         execution-trace-total-ms
         set-execution-trace-prefill-ms!
         set-execution-trace-decode-ms!
         set-execution-trace-total-ms!
         trace-begin-step!
         trace-activation!
         trace-kv!
         trace-attention!
         trace-bpe-chunk!
         trace-span!
         trace-logits!
         trace-sample!
         trace-schedule!
         execution-trace->jsexpr
         write-execution-trace
         if-trace
         with-trace-phase
         tensor-summary
         activation-histogram
         attention-quant-scale
         attention-quant-decimals)

(define current-trace-observer (make-parameter #f))
(define current-trace-stage (make-parameter #f))
(define current-trace-layer (make-parameter #f))
(define current-trace-phase (make-parameter #f))
;; When not #f, a box filled with the pre-residual matmul tensor.
;; matmul-add reads it. The default #f does not copy.
(define current-trace-capture (make-parameter #f))

(struct execution-trace (bpe steps schedules timings attention
                          prefill-ms decode-ms total-ms)
  #:mutable
  #:constructor-name make-trace-state)

(struct trace-step (phase input-ids root blocks) #:transparent)

(define (make-execution-trace)
  (make-trace-state '() '() '() '() '() #f #f #f))

(define attention-quant-scale 10000)
(define attention-quant-decimals 4)
(define histogram-bins 32)

(define-syntax-rule (if-trace obs stage-id expr on-trace)
  (if (eq? obs #f)
      expr
      (let ([t0 (current-inexact-monotonic-milliseconds)])
        (define y (parameterize ([current-trace-stage stage-id])
                    expr))
        (on-trace y t0)
        y)))

(define-syntax-rule (with-trace-phase obs phase ids . body)
  (if (eq? obs #f)
      (let () . body)
      (parameterize ([current-trace-phase phase])
        (trace-begin-step! obs phase ids)
        (let () . body))))

(define (trace-begin-step! obs phase input-ids)
  (set-execution-trace-steps!
   obs
   (cons (trace-step phase input-ids (make-hasheq) (make-hasheq))
         (execution-trace-steps obs))))

(define (open-step obs)
  (define steps (execution-trace-steps obs))
  (if (null? steps)
      (error 'trace "recording outside a forward step")
      (car steps)))

(define (block-table step layer)
  (define blocks (trace-step-blocks step))
  (or (hash-ref blocks layer #f)
      (let ([table (make-hasheq)])
        (hash-set! blocks layer table)
        table)))

(define (put-stage! obs stage value)
  (define step (open-step obs))
  (define layer (current-trace-layer))
  (if (eq? layer #f)
      (hash-set! (trace-step-root step) stage value)
      (hash-set! (block-table step layer) stage value)))

(define (step-index obs)
  (define n (length (execution-trace-steps obs)))
  (if (zero? n) #f (sub1 n)))

(define (push-timing! obs stage layer phase ms)
  (set-execution-trace-timings!
   obs
   (cons (hasheq 'stage stage
                 'layer layer
                 'phase phase
                 'step (step-index obs)
                 'ms ms)
         (execution-trace-timings obs))))

(define (tensor-summary t)
  (define data (tensor-data t))
  (define n (flvector-length data))
  (when (zero? n)
    (error 'tensor-summary "empty tensor"))
  (define-values (minv maxv sum sumsq)
    (for/fold ([minv +inf.0] [maxv -inf.0] [sum 0.0] [sumsq 0.0])
              ([i (in-range n)])
      (define x (flvector-ref data i))
      (values (flmin minv x)
              (flmax maxv x)
              (fl+ sum x)
              (fl+ sumsq (fl* x x)))))
  (define inv (fl/ 1.0 (real->double-flonum n)))
  (define mean (fl* sum inv))
  (define var (fl- (fl* sumsq inv) (fl* mean mean)))
  (define std (flsqrt (if (fl> var 0.0) var 0.0)))
  (hasheq 'shape (tensor-dims t)
          'min minv
          'max maxv
          'mean mean
          'std std
          'l2 (flsqrt sumsq)))

(define (activation-histogram data bins)
  (define n (flvector-length data))
  (define-values (minv maxv)
    (for/fold ([minv +inf.0] [maxv -inf.0])
              ([i (in-range n)])
      (define x (flvector-ref data i))
      (values (flmin minv x) (flmax maxv x))))
  (define counts (make-vector bins 0))
  (define span (fl- maxv minv))
  (define bins-fl (real->double-flonum bins))
  (for ([i (in-range n)])
    (define x (flvector-ref data i))
    (define b
      (if (fl<= span 0.0)
          0
          (let ([raw (fl* (fl/ (fl- x minv) span) bins-fl)])
            (if (fl>= raw bins-fl)
                (sub1 bins)
                (inexact->exact (flfloor raw))))))
    (vector-set! counts b (add1 (vector-ref counts b))))
  (hasheq 'bins bins
          'min minv
          'max maxv
          'quantization "uniform"
          'counts (vector->list counts)))

(define (trace-activation! obs stage tensor t0)
  (define entry
    (hasheq 'summary (tensor-summary tensor)
            'histogram (activation-histogram (tensor-data tensor) histogram-bins)))
  (put-stage! obs stage entry)
  (unless (eq? t0 #f)
    (push-timing! obs stage (current-trace-layer) (current-trace-phase)
                  (- (current-inexact-monotonic-milliseconds) t0))))

(define (trace-kv! obs info t0)
  (put-stage! obs 'kv-write info)
  (push-timing! obs 'kv-write (current-trace-layer) (current-trace-phase)
                (- (current-inexact-monotonic-milliseconds) t0)))

(define (trace-attention! obs slots)
  (define layer (current-trace-layer))
  (define acc (execution-trace-attention obs))
  (for ([h (in-range (vector-length slots))])
    (for ([item (in-list (vector-ref slots h))])
      (set! acc (cons (list layer h (car item) (cdr item)) acc))))
  (set-execution-trace-attention! obs acc))

(define (trace-bpe-chunk! obs chunk)
  (set-execution-trace-bpe!
   obs
   (cons chunk (execution-trace-bpe obs))))

(define (trace-span! obs stage layer phase ms)
  (push-timing! obs stage layer phase ms))

(define (trace-logits! obs top)
  (hash-set! (trace-step-root (open-step obs)) 'tied-logits-top top))

(define (trace-sample! obs #:decoding decoding #:id id #:text text
                       #:cumulative cumulative #:ms ms)
  (hash-set! (trace-step-root (open-step obs))
             'sample
             (hasheq 'decoding decoding
                     'id id
                     'text text
                     'cumulative cumulative))
  (push-timing! obs 'sample #f (current-trace-phase) ms))

(define (trace-schedule! obs rec)
  (set-execution-trace-schedules!
   obs
   (cons rec (execution-trace-schedules obs))))

(define (json-null v)
  (if (eq? v #f) 'null v))

(define (phase-name v)
  (cond
    [(symbol? v) (symbol->string v)]
    [(string? v) v]
    [else 'null]))

(define (stage-name v)
  (cond
    [(symbol? v) (symbol->string v)]
    [(string? v) v]
    [else 'null]))

(define (summary-jsexpr summary)
  (hasheq 'shape (hash-ref summary 'shape)
          'min (hash-ref summary 'min)
          'max (hash-ref summary 'max)
          'mean (hash-ref summary 'mean)
          'std (hash-ref summary 'std)
          'l2 (hash-ref summary 'l2)))

(define (push-histogram! acc stage layer phase step entry)
  (define hist (hash-ref entry 'histogram))
  (cons (hasheq 'stage (stage-name stage)
                'layer (json-null layer)
                'phase (phase-name phase)
                'step (json-null step)
                'bins (hash-ref hist 'bins)
                'min (hash-ref hist 'min)
                'max (hash-ref hist 'max)
                'quantization (hash-ref hist 'quantization)
                'counts (hash-ref hist 'counts))
        acc))

;; Prefill and the final forward keep every block activation. Every step
;; keeps the residual-stream ends (block-ln1, block-mlp-res), the KV write,
;; and the root tensors. The other per-layer decode activations are timed
;; and scheduled, and their attention weights are kept, but a second copy
;; of those summary floats does not fit the trace size budget.
(define (export-step step index nsteps histograms)
  (define full? (or (zero? index) (= index (sub1 nsteps))))
  (define root (trace-step-root step))
  (define phase (trace-step-phase step))
  (define (take-root! name)
    (define entry (hash-ref root name #f))
    (and entry
         (begin
           (set! histograms (push-histogram! histograms name #f phase index entry))
           (summary-jsexpr (hash-ref entry 'summary)))))
  (define blocks
    (for/list ([layer (in-list (sort (hash-keys (trace-step-blocks step)) <))])
      (define table (hash-ref (trace-step-blocks step) layer))
      (define (take! name)
        (define entry (hash-ref table name #f))
        (and entry
             (or full? (memq name '(block-ln1 block-mlp-res)))
             (begin
               (set! histograms (push-histogram! histograms name layer phase index entry))
               (summary-jsexpr (hash-ref entry 'summary)))))
      (define mlp (take! 'mlp-gelu))
      (define proj (take! 'mlp-cproj))
      (define body (make-hasheq))
      (hash-set! body 'layer layer)
      (define ln1 (take! 'block-ln1))
      (when ln1 (hash-set! body 'block-ln1 ln1))
      (define qkv (take! 'block-qkv))
      (when qkv (hash-set! body 'block-qkv qkv))
      (define kv (hash-ref table 'kv-write #f))
      (when kv (hash-set! body 'kv-write kv))
      (define attend (take! 'attend))
      (when attend (hash-set! body 'attend attend))
      (define proj-res (take! 'block-proj-res))
      (when proj-res (hash-set! body 'block-proj-res proj-res))
      (define ln2 (take! 'block-ln2))
      (when ln2 (hash-set! body 'block-ln2 ln2))
      (when mlp
        (hash-set! body 'mlp-gelu
                   (if proj (hash-set mlp 'cProj proj) mlp)))
      (define mlp-res (take! 'block-mlp-res))
      (when mlp-res (hash-set! body 'block-mlp-res mlp-res))
      body))
  (define logits (take-root! 'tied-logits))
  (define top (hash-ref root 'tied-logits-top #f))
  (define tied
    (cond
      [(and logits top) (hash-set logits 'top top)]
      [logits logits]
      [top (hasheq 'top top)]
      [else #f]))
  (define out (make-hasheq))
  (hash-set! out 'phase (phase-name phase))
  (hash-set! out 'inputIds (trace-step-input-ids step))
  (define embed (take-root! 'input-embed))
  (when embed (hash-set! out 'input-embed embed))
  (hash-set! out 'blocks blocks)
  (define final (take-root! 'final-ln))
  (when final (hash-set! out 'final-ln final))
  (when tied (hash-set! out 'tied-logits tied))
  (define sample (hash-ref root 'sample #f))
  (when sample (hash-set! out 'sample sample))
  (values out histograms))

(define (flvector-sum data)
  (for/fold ([s 0.0]) ([i (in-range (flvector-length data))])
    (fl+ s (flvector-ref data i))))

(define (quantize-weight w)
  (define scaled (round (fl* w (real->double-flonum attention-quant-scale))))
  (define n (inexact->exact scaled))
  (cond
    [(< n 0) 0]
    [(> n attention-quant-scale) attention-quant-scale]
    [else n]))

(define (attention-jsexpr rows)
  (define present (make-hash))
  (define max-layer -1)
  (define max-head -1)
  (define max-row -1)
  (for ([item (in-list rows)])
    (define layer (car item))
    (define head (cadr item))
    (define row (caddr item))
    (define vec (cadddr item))
    (hash-set! present (list layer head row) vec)
    (when (> layer max-layer) (set! max-layer layer))
    (when (> head max-head) (set! max-head head))
    (when (> row max-row) (set! max-row row)))
  (define n-layer (add1 max-layer))
  (define n-head (add1 max-head))
  (define n-row (add1 max-row))
  (define acc '())
  (define exact-err 0.0)
  (define quant-err 0.0)
  (define pairs 0)
  (for ([layer (in-range n-layer)])
    (for ([head (in-range n-head)])
      (for ([row (in-range n-row)])
        (define vec
          (hash-ref present (list layer head row)
                    (lambda ()
                      (error 'trace "missing attention row ~a ~a ~a" layer head row))))
        (unless (= (flvector-length vec) (add1 row))
          (error 'trace "attention row ~a has ~a weights, expected ~a"
                 row (flvector-length vec) (add1 row)))
        (set! pairs (+ pairs (flvector-length vec)))
        (define sum (flvector-sum vec))
        (define err (flabs (fl- sum 1.0)))
        (when (fl> err exact-err) (set! exact-err err))
        (define qsum 0)
        (for ([j (in-range (flvector-length vec))])
          (define q (quantize-weight (flvector-ref vec j)))
          (set! qsum (+ qsum q))
          (set! acc (cons q acc)))
        (define qerr
          (flabs (fl- (fl/ (real->double-flonum qsum)
                           (real->double-flonum attention-quant-scale))
                      1.0)))
        (when (fl> qerr quant-err) (set! quant-err qerr)))))
  (hasheq 'encoding "scaled-int"
          'scale attention-quant-scale
          'decimals attention-quant-decimals
          'maxRowSumError quant-err
          'exactMaxRowSumError exact-err
          'layers n-layer
          'heads n-head
          'queryRows n-row
          'pairs pairs
          'order '("layer" "head" "query" "key")
          'causal #t
          'values (reverse acc)))

(define (round-ms ms)
  ;; Microsecond display of a measured millisecond reading. The value is
  ;; the clock sample, not a synthesized duration.
  (/ (round (* (real->double-flonum ms) 1000.0)) 1000.0))

(define (timing-jsexpr rec)
  (hasheq 'stage (stage-name (hash-ref rec 'stage))
          'layer (json-null (hash-ref rec 'layer))
          'phase (phase-name (hash-ref rec 'phase))
          'step (json-null (hash-ref rec 'step))
          'ms (round-ms (hash-ref rec 'ms))))

(define (source-map-jsexpr sha)
  (for/hasheq ([id (in-list stage-ids)])
    (define info
      (hash-ref stage-source-table id
                (lambda () (error 'source-map "missing stage ~a" id))))
    (values id
            (hasheq 'commitSha sha
                    'file (hash-ref info 'file)
                    'line (hash-ref info 'line)
                    'endLine (hash-ref info 'endLine)
                    'fn (hash-ref info 'fn)
                    'snippet (hash-ref info 'snippet)))))

(define (execution-trace->jsexpr tr #:commit-sha sha #:model model #:run run)
  (define histograms '())
  (define ordered (reverse (execution-trace-steps tr)))
  (define nsteps (length ordered))
  (define steps
    (for/list ([step (in-list ordered)]
               [index (in-naturals)])
      (define-values (exported hist) (export-step step index nsteps histograms))
      (set! histograms hist)
      exported))
  (define attention
    (and (pair? (execution-trace-attention tr))
         (attention-jsexpr (execution-trace-attention tr))))
  (hasheq 'schemaVersion "1.0.0"
          'commitSha sha
          'model model
          'run run
          'sourceMap (source-map-jsexpr sha)
          'coverage
          (hasheq 'fullBlockSteps "prefill and the final forward"
                  'everyStep '("input-embed" "block-ln1" "kv-write"
                               "block-mlp-res" "final-ln" "tied-logits")
                  'attention "every causal pair"
                  'scheduling "every matmul policy decision"
                  'timings "measured stage spans, milliseconds rounded to 0.001")
          'exact (hasheq 'bpe (hasheq 'chunks (reverse (execution-trace-bpe tr)))
                         'steps steps
                         'attention
                         (if attention
                             (hasheq 'pairs (hash-ref attention 'pairs)
                                     'maxRowSumError (hash-ref attention 'exactMaxRowSumError))
                             'null))
          'quantized (hasheq 'attention (or attention 'null)
                             'histograms
                             (hasheq 'bins histogram-bins
                                     'quantization "uniform"
                                     'items (reverse histograms)))
          'scheduling (reverse (execution-trace-schedules tr))
          'timings (hasheq 'clock "current-inexact-monotonic-milliseconds"
                           'prefillMs (json-null (execution-trace-prefill-ms tr))
                           'decodeMs (json-null (execution-trace-decode-ms tr))
                           'totalMs (json-null (execution-trace-total-ms tr))
                           'stages (map timing-jsexpr (reverse (execution-trace-timings tr))))))

(define (write-execution-trace path jsexpr)
  (call-with-output-file path
    (lambda (out)
      (write-json jsexpr out)
      (newline out))
    #:exists 'replace))
