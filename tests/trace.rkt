#lang racket/base

;; Execution-trace observer, source map, and schema.
;; The 124M section runs only when the pinned weights are present.

(module+ test
  (require json
           racket/file
           racket/flonum
           racket/future
           racket/list
           racket/port
           racket/runtime-path
           racket/string
           racket/system
           rackunit
           "../src/attention.rkt"
           "../src/model.rkt"
           "../src/nn.rkt"
           "../src/source-map.rkt"
           "../src/tensor.rkt"
           "../src/tokenizer.rkt"
           "../src/trace-run.rkt"
           "../src/trace.rkt")

  (define-runtime-path repo-root "..")
  (define-runtime-path tiny-dir "tiny-gpt2")
  (define-runtime-path gpt2-dir "../models/gpt2")

  (define (file-text rel)
    (file->string (build-path repo-root rel)))

  (define (file-line rel n)
    (list-ref (string-split (file-text rel) "\n") (sub1 n)))

  (define (flvector-max-abs-diff a b)
    (for/fold ([m 0.0]) ([i (in-range (flvector-length a))])
      (flmax m (flabs (fl- (flvector-ref a i) (flvector-ref b i))))))

  (define (argmax-row logits)
    (define data (tensor-data logits))
    (define n (flvector-length data))
    (let loop ([i 1] [best 0] [bestv (flvector-ref data 0)])
      (if (= i n)
          best
          (let ([v (flvector-ref data i)])
            (if (fl> v bestv)
                (loop (add1 i) i v)
                (loop (add1 i) best bestv))))))

  (define (collect-logits model ids n)
    (define cache (empty-kv-cache model #:capacity (+ (length ids) n)))
    (define-values (logits cache2)
      (model-forward model ids cache #:last-only #t))
    (let loop ([cache cache2]
               [prev (argmax-row logits)]
               [left (sub1 n)]
               [rows (list (flvector-copy (tensor-data logits)))]
               [chosen (list (argmax-row logits))])
      (if (zero? left)
          (values (reverse rows) (reverse chosen))
          (let-values ([(logits cache) (model-forward model (list prev) cache #:last-only #t)])
            (define id (argmax-row logits))
            (loop cache id (sub1 left)
                  (cons (flvector-copy (tensor-data logits)) rows)
                  (cons id chosen))))))

  (define (ones rows cols)
    (make-tensor (list rows cols) (make-flvector (* rows cols) 1.0)))

  (define (zeros n)
    (make-tensor (list n) (make-flvector n 0.0)))

  (test-case "source map pins all 13 stages to repo files"
    (check-true (source-map-complete?))
    (for ([id (in-list stage-ids)])
      (define info (hash-ref stage-source-table id))
      (define file (hash-ref info 'file))
      (define line (hash-ref info 'line))
      (define end-line (hash-ref info 'endLine))
      (define snippet (hash-ref info 'snippet))
      (check-regexp-match #px"^src/[^/]+\\.rkt$" file)
      (check-false (string-prefix? file "/"))
      (check-true (exact-positive-integer? line))
      (check-true (and (exact-positive-integer? end-line) (>= end-line line)))
      (check-true (non-empty-string? (hash-ref info 'fn)))
      (check-true (non-empty-string? snippet))
      (check-true (string-prefix? (string-trim (file-line file line)) snippet))
      (check-true (string-contains? (file-text file) snippet)))
    (void))

  (test-case "BPE trace records merges and matches encode"
    (define tok
      (make-tokenizer (hash "a" 0 "b" 1 "ab" 2 "c" 3)
                      (hash (cons "a" "b") 0)))
    (define obs (make-execution-trace))
    (define traced
      (parameterize ([current-trace-observer obs])
        (append (encode tok "abc")
                (encode tok "ab")
                (encode tok "ab"))))
    (define cold (make-tokenizer (hash "a" 0 "b" 1 "ab" 2 "c" 3)
                                 (hash (cons "a" "b") 0)))
    (check-equal? traced (append (encode cold "abc") (encode cold "ab") (encode cold "ab")))
    (define doc
      (execution-trace->jsexpr
       obs
       #:commit-sha (git-commit-sha)
       #:model (hasheq 'repo "openai-community/gpt2"
                       'revision "607a30d783dfa663caf39e06633721c8d4cfcd7e"
                       'nLayer 0 'nHead 1 'nEmbd 1 'vocabSize 4 'nCtx 8)
       #:run (hasheq 'prompt "abc" 'maxNewTokens 1 'decoding "greedy"
                     'promptTokenIds traced 'generatedTokenIds '()
                     'allTokenIds traced 'tokens '() 'outputText "")))
    (define chunks (hash-ref (hash-ref (hash-ref doc 'exact) 'bpe) 'chunks))
    (check-true (pair? chunks))
    (define merged (findf (lambda (chunk) (pair? (hash-ref chunk 'merges '()))) chunks))
    (check-not-false merged)
    (define step (car (hash-ref merged 'merges)))
    (check-equal? (hash-ref step 'pair) '("a" "b"))
    (check-equal? (hash-ref step 'rank) 0)
    (check-equal? (hash-ref step 'merged) "ab")
    (check-equal? (hash-ref step 'pieces) '("ab" "c"))
    (check-equal? (hash-ref merged 'tokenIds) '(2 3))
    (check-false (hash-ref merged 'cached))
    (define cached (findf (lambda (chunk) (eq? (hash-ref chunk 'cached) #t)) chunks))
    (check-not-false cached)
    (check-eq? (hash-ref cached 'merges) 'null)
    (check-equal? (hash-ref cached 'tokenIds) '(2))
    (void))

  (test-case "attention weights-out matches the fast path and is causal"
    (define q (make-tensor '(3 4) (flvector 1.0 0.0 0.0 0.0
                                            0.0 1.0 0.0 0.0
                                            0.5 0.5 0.0 0.0)))
    (define k (make-tensor '(3 4) (flvector 1.0 0.0 0.0 0.0
                                            0.0 1.0 0.0 0.0
                                            1.0 1.0 0.0 0.0)))
    (define v (make-tensor '(3 4) (flvector 1.0 2.0 3.0 4.0
                                            5.0 6.0 7.0 8.0
                                            9.0 8.0 7.0 6.0)))
    (define fast (attend q k v 2))
    (define slots (make-vector 2 '()))
    (define recorded (attend q k v 2 #:weights-out slots))
    (check-equal? (tensor-data fast) (tensor-data recorded))
    (for ([h (in-range 2)])
      (define rows (vector-ref slots h))
      (check-equal? (length rows) 3)
      (for ([item (in-list rows)] [expect (in-naturals)])
        (check-equal? (car item) expect)
        (define w (cdr item))
        (check-equal? (flvector-length w) (add1 expect))
        (define sum (for/fold ([s 0.0]) ([i (in-range (flvector-length w))])
                      (fl+ s (flvector-ref w i))))
        (check-= sum 1.0 1e-12)))
    (void))

  (test-case "matmul schedule is metadata and timings are separate"
    (define obs (make-execution-trace))
    (parameterize ([current-trace-observer obs]
                   [current-trace-stage 'block-qkv]
                   [current-trace-phase 'prefill]
                   [current-trace-layer #f])
      (matmul (ones 2 2) (ones 2 2) (zeros 2)))
    (parameterize ([current-trace-observer obs]
                   [current-trace-stage 'tied-logits]
                   [current-trace-phase 'decode]
                   [current-trace-layer 3])
      (matmul (ones 32 64) (ones 64 64) (zeros 64))
      (matmul (ones 1 200) (ones 200 600) (zeros 600)))
    (define doc
      (execution-trace->jsexpr
       obs
       #:commit-sha "0123456789abcdef0123456789abcdef01234567"
       #:model (hasheq 'repo "openai-community/gpt2"
                       'revision "607a30d783dfa663caf39e06633721c8d4cfcd7e"
                       'nLayer 1 'nHead 1 'nEmbd 2 'vocabSize 2 'nCtx 8)
       #:run (hasheq 'prompt "" 'maxNewTokens 1 'decoding "greedy"
                     'promptTokenIds '() 'generatedTokenIds '()
                     'allTokenIds '() 'tokens '() 'outputText "")))
    (define rows (hash-ref doc 'scheduling))
    (check-equal? (length rows) 3)
    (define serial (car rows))
    (check-equal? (hash-ref serial 'axis) "serial")
    (check-equal? (hash-ref serial 'ops) 8)
    (check-equal? (hash-ref serial 'splitCount) 1)
    (check-equal? (hash-ref (hash-ref serial 'thresholds) 'work) 100000)
    (check-equal? (hash-ref (hash-ref serial 'thresholds) 'axis) 32)
    (check-equal? (hash-ref serial 'processorCount) (processor-count))
    (check-false (hash-has-key? serial 'ms))
    (define row-split (cadr rows))
    (check-equal? (hash-ref row-split 'axis) "row")
    (check-equal? (hash-ref row-split 'ops) (* 32 64 64))
    (check-true (> (hash-ref row-split 'splitCount) 1))
    (define covered
      (for/sum ([chunk (in-list (hash-ref row-split 'chunks))])
        (- (hash-ref chunk 'i1) (hash-ref chunk 'i0))))
    (check-equal? covered 32)
    (define col-split (caddr rows))
    (check-equal? (hash-ref col-split 'axis) "col")
    (check-equal? (hash-ref col-split 'layer) 3)
    (check-equal? (hash-ref col-split 'phase) "decode")
    (check-false (hash-has-key? col-split 'ms))
    (define off (matmul (ones 32 64) (ones 64 64) (zeros 64)))
    (define on
      (parameterize ([current-trace-observer (make-execution-trace)])
        (matmul (ones 32 64) (ones 64 64) (zeros 64))))
    (check-equal? (tensor-data off) (tensor-data on))
    (void))

  (define tiny (load-model tiny-dir))

  (test-case "tiny greedy logits and token ids match with tracing on"
    (define ids (encode (gpt2-model-tokenizer tiny) "ab"))
    (define-values (off-rows off-ids)
      (parameterize ([current-trace-observer #f])
        (collect-logits tiny ids 3)))
    (define trace (make-execution-trace))
    (define-values (on-rows on-ids)
      (parameterize ([current-trace-observer trace])
        (collect-logits tiny ids 3)))
    (check-equal? on-ids off-ids)
    (for ([a (in-list on-rows)] [b (in-list off-rows)])
      (check-equal? (flvector-max-abs-diff a b) 0.0)
      (check-equal? a b))
    (define generated
      (parameterize ([current-trace-observer #f])
        (generation-token-ids (model-generate tiny ids 3))))
    (define generated-on
      (parameterize ([current-trace-observer (make-execution-trace)])
        (generation-token-ids (model-generate tiny ids 3))))
    (check-equal? generated-on generated)
    (check-true (activation-trace? (model-activations tiny ids)))
    (void))

  (test-case "tiny traced generation has the schema and causal rows"
    (define-values (doc result ids)
      (trace-generation tiny "ab" 3 #:commit-sha (git-commit-sha)))
    (check-equal? (hash-ref doc 'schemaVersion) "1.0.0")
    (check-equal? (hash-ref doc 'commitSha) (git-commit-sha))
    (define source (hash-ref doc 'sourceMap))
    (for ([id (in-list stage-ids)])
      (define entry (hash-ref source id))
      (check-equal? (hash-ref entry 'commitSha) (hash-ref doc 'commitSha))
      (check-regexp-match #px"^src/[^/]+\\.rkt$" (hash-ref entry 'file)))
    (define steps (hash-ref (hash-ref doc 'exact) 'steps))
    (define sampled (filter (lambda (step) (hash-ref step 'sample #f)) steps))
    (check-equal? (length sampled) 3)
    (check-equal? (hash-ref (car steps) 'phase) "prefill")
    (check-equal? (map (lambda (step) (hash-ref (hash-ref step 'sample) 'id)) sampled)
                  (generation-token-ids result))
    (define attn (hash-ref (hash-ref doc 'quantized) 'attention))
    (check-equal? (hash-ref attn 'encoding) "scaled-int")
    (check-equal? (hash-ref attn 'scale) 10000)
    (check-equal? (hash-ref attn 'decimals) 4)
    (check-equal? (hash-ref attn 'causal) #t)
    (check-equal? (hash-ref attn 'queryRows) (+ (length ids) 3))
    (check-equal? (hash-ref attn 'pairs)
                  (* (hash-ref attn 'layers)
                     (hash-ref attn 'heads)
                     (quotient (* (hash-ref attn 'queryRows)
                                  (add1 (hash-ref attn 'queryRows)))
                               2)))
    (check-equal? (length (hash-ref attn 'values)) (hash-ref attn 'pairs))
    (check-true (< (hash-ref attn 'exactMaxRowSumError) 1e-9))
    (check-true (< (hash-ref attn 'maxRowSumError) 1e-3))
    (define hists (hash-ref (hash-ref (hash-ref doc 'quantized) 'histograms) 'items))
    (check-true (pair? hists))
    (check-equal? (hash-ref (car hists) 'bins) 32)
    (check-equal? (hash-ref (car hists) 'quantization) "uniform")
    (check-equal? (length (hash-ref (car hists) 'counts)) 32)
    (define timings (hash-ref (hash-ref doc 'timings) 'stages))
    (check-true (pair? timings))
    (check-false (ormap (lambda (row) (hash-has-key? row 'chunks)) timings))
    (check-equal? (hash-ref (hash-ref doc 'timings) 'clock)
                  "current-inexact-monotonic-milliseconds")
    (check-true (>= (hash-ref (hash-ref doc 'timings) 'totalMs) 0))
    (define middle (cadr steps))
    (check-true (hash-has-key? (car (hash-ref middle 'blocks)) 'block-ln1))
    (check-true (hash-has-key? (car (hash-ref middle 'blocks)) 'block-mlp-res))
    (check-false (hash-has-key? (car (hash-ref middle 'blocks)) 'block-qkv))
    (check-true (hash-has-key? (car (hash-ref (last steps) 'blocks)) 'block-qkv))
    (define kv (hash-ref (car (hash-ref (car steps) 'blocks)) 'kv-write))
    (check-equal? (hash-ref kv 'filledBefore) 0)
    (check-equal? (hash-ref kv 'zeroCopy) #t)
    (check-equal? (hash-ref kv 'elementBytes) 8)
    (check-true (positive? (hash-ref kv 'strideBytes)))
    (check-true (positive? (hash-ref kv 'bytesWritten)))
    (void))

  (define (weights-ready?)
    (file-exists? (build-path gpt2-dir "model.safetensors")))

  (define (gzip-size path)
    (define out (open-output-bytes))
    (parameterize ([current-output-port out]
                   [current-error-port (open-output-nowhere)])
      (unless (system* (find-executable-path "gzip") "-9" "-c" (path->string path))
        (error 'gzip "gzip failed")))
    (bytes-length (get-output-bytes out)))

  (cond
    [(not (weights-ready?))
     (printf "skipping 124M execution trace: ~a has no weights\n" gpt2-dir)]
    [else
     (define gpt2 (load-model gpt2-dir))
     (define prompt "Once upon a time")
     (define expect-ids '(11 262 995 373 257 1295 286 1049 8737 290 1049 3514 13 383 995 373))
     (define expect-text ", the world was a place of great beauty and great danger. The world was")

     (test-case "124M greedy logits match exactly with tracing on"
       (define ids (encode (gpt2-model-tokenizer gpt2) prompt))
       (check-equal? ids '(7454 2402 257 640))
       (define-values (off-rows off-ids)
         (parameterize ([current-trace-observer #f])
           (collect-logits gpt2 ids 16)))
       (define-values (on-rows on-ids)
         (parameterize ([current-trace-observer (make-execution-trace)])
           (collect-logits gpt2 ids 16)))
       (check-equal? off-ids expect-ids)
       (check-equal? on-ids off-ids)
       (for ([a (in-list on-rows)] [b (in-list off-rows)])
         (check-equal? (flvector-length a) 50257)
         (check-equal? (flvector-max-abs-diff a b) 0.0)
         (check-equal? a b))
       (printf "124M parity max-abs 0.0 across ~a logit rows\n" (length on-rows))
       (void))

     (test-case "124M trace has 30240 causal pairs, row sums, and a small gzip"
       (define-values (doc result ids)
         (trace-generation gpt2 prompt 16))
       (check-equal? ids '(7454 2402 257 640))
       (check-equal? (generation-token-ids result) expect-ids)
       (check-equal? (generation-text result) expect-text)
       (check-equal? (hash-ref (hash-ref doc 'run) 'outputText) expect-text)
       (check-equal? (hash-ref (hash-ref doc 'run) 'allTokenIds) (append ids expect-ids))
       (check-equal? (hash-ref (hash-ref doc 'model) 'nLayer) 12)
       (check-equal? (hash-ref (hash-ref doc 'model) 'nHead) 12)
       (check-equal? (hash-ref (hash-ref doc 'model) 'nEmbd) 768)
       (check-equal? (hash-ref (hash-ref doc 'model) 'vocabSize) 50257)
       (check-equal? (hash-ref (hash-ref doc 'model) 'nCtx) 1024)
       (check-equal? (hash-ref (hash-ref doc 'model) 'repo) "openai-community/gpt2")
       (check-equal? (hash-ref (hash-ref doc 'model) 'revision)
                     "607a30d783dfa663caf39e06633721c8d4cfcd7e")
       (define attn (hash-ref (hash-ref doc 'quantized) 'attention))
       (check-equal? (hash-ref attn 'layers) 12)
       (check-equal? (hash-ref attn 'heads) 12)
       (check-equal? (hash-ref attn 'queryRows) 20)
       (check-equal? (hash-ref attn 'pairs) 30240)
       (check-equal? (length (hash-ref attn 'values)) 30240)
       (define values-rev (hash-ref attn 'values))
       (define cursor 0)
       (for ([layer (in-range 12)])
         (for ([head (in-range 12)])
           (for ([row (in-range 20)])
             (define width (add1 row))
             (define slice (take (drop values-rev cursor) width))
             (check-true (andmap (lambda (q) (and (exact-integer? q)
                                                  (<= 0 q (hash-ref attn 'scale))))
                                 slice))
             (set! cursor (+ cursor width)))))
       (check-equal? cursor 30240)
       (check-true (< (hash-ref attn 'exactMaxRowSumError) 1e-9))
       (check-true (< (hash-ref attn 'maxRowSumError) 2e-3))
       (define all-steps (hash-ref (hash-ref doc 'exact) 'steps))
       (define prefill (car all-steps))
       (define middle (cadr all-steps))
       (define final (last all-steps))
       (check-true (hash-has-key? (car (hash-ref prefill 'blocks)) 'block-qkv))
       (check-true (hash-has-key? (car (hash-ref prefill 'blocks)) 'attend))
       (check-true (hash-has-key? (car (hash-ref middle 'blocks)) 'block-ln1))
       (check-true (hash-has-key? (car (hash-ref middle 'blocks)) 'block-mlp-res))
       (check-true (hash-has-key? (car (hash-ref middle 'blocks)) 'kv-write))
       (check-false (hash-has-key? (car (hash-ref middle 'blocks)) 'block-qkv))
       (check-true (hash-has-key? (car (hash-ref final 'blocks)) 'mlp-gelu))
       (define sampled
         (filter (lambda (step) (hash-ref step 'sample #f))
                 all-steps))
       (check-equal? (length sampled) 16)
       (check-equal? (length (hash-ref (hash-ref (car sampled) 'tied-logits) 'top)) 15)
       (define top0 (car (hash-ref (hash-ref (car sampled) 'tied-logits) 'top)))
       (check-equal? (hash-ref top0 'id) 11)
       (define path (make-temporary-file "gpt2-trace-~a.json"))
       (write-execution-trace path doc)
       (define raw (file-size path))
       (define gz (gzip-size path))
       (printf "124M trace raw ~a gzip ~a exact-row-sum-error ~a quant-row-sum-error ~a\n"
               raw gz
               (hash-ref attn 'exactMaxRowSumError)
               (hash-ref attn 'maxRowSumError))
       (check-true (<= gz 153600))
       (delete-file path)
       (void))]))
