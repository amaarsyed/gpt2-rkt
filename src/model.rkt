#lang racket/base

;; Full GPT-2 forward pass.
;; wte + wpe, then the pre-LN blocks, then ln_f, then logits = h · wteᵀ.
;; The tied head uses matmul-transposed-b, so wte stays stored [vocab, d].
;; A step may append one token: each layer keeps the key and value rows
;; it has already computed, and position-embed starts at the cache length.

(require json
         racket/flonum
         racket/list
         "block.rkt"
         "embed.rkt"
         "errors.rkt"
         "model-assets.rkt"
         "nn.rkt"
         "safetensors.rkt"
         "tensor.rkt"
         "tokenizer.rkt")

(provide load-model
         make-gpt2
         gpt2-model?
         gpt2-model-vocab-size
         gpt2-model-n-positions
         gpt2-model-eos-id
         gpt2-model-tokenizer
         next-token-logits
         model-forward
         empty-kv-cache
         generation?
         generation-token-ids
         generation-text
         model-generate)

(struct gpt2-model (vocab-size n-positions n-embd n-head eos-id
                    wte wpe blocks ln-gain ln-bias tokenizer)
  #:constructor-name raw-gpt2
  #:transparent)

(struct kv-cache (layers position) #:transparent)

(struct generation (token-ids text) #:transparent)

(define (make-gpt2 #:wte wte
                   #:wpe wpe
                   #:blocks blocks
                   #:ln-gain ln-gain
                   #:ln-bias ln-bias
                   #:eos-id eos-id
                   #:tokenizer [tokenizer #f])
  (unless (and (tensor? wte) (= (length (tensor-dims wte)) 2))
    (raise-argument-error 'make-gpt2 "wte tensor?" wte))
  (unless (list? blocks)
    (raise-argument-error 'make-gpt2 "list?" blocks))
  (raw-gpt2 (car (tensor-dims wte))
            (car (tensor-dims wpe))
            (cadr (tensor-dims wte))
            (if (null? blocks) 1 (block-params-n-head (car blocks)))
            eos-id
            wte wpe blocks ln-gain ln-bias tokenizer))

(define (empty-kv-cache model)
  (kv-cache (make-list (length (gpt2-model-blocks model)) #f) 0))

(define (context-limit model)
  (min gpt2-context-length (gpt2-model-n-positions model)))

(define (check-context who n limit)
  (when (> n limit)
    (raise-gpt2
     (format "~a: sequence is longer than the context: ~a tokens, limit ~a"
             who n limit))))

(define (tied-logits h wte)
  (matmul-transposed-b h wte))

(define (model-forward model ids cache)
  (unless (gpt2-model? model)
    (raise-argument-error 'model-forward "gpt2-model?" model))
  (unless (and (list? ids) (andmap exact-nonnegative-integer? ids))
    (raise-argument-error 'model-forward "list of token ids?" ids))
  (define start (kv-cache-position cache))
  (define n (length ids))
  (check-context 'model-forward (+ start n) (context-limit model))
  (define x (input-embed (gpt2-model-wte model)
                         (gpt2-model-wpe model)
                         ids
                         #:start start))
  (define-values (y layers)
    (for/fold ([hidden x] [acc '()])
              ([params (in-list (gpt2-model-blocks model))]
               [past (in-list (kv-cache-layers cache))])
      (define-values (out k v)
        (gpt2-block-cached hidden params (and past (car past)) (and past (cdr past))))
      (values out (cons (cons k v) acc))))
  (define hidden (layer-norm y (gpt2-model-ln-gain model) (gpt2-model-ln-bias model)))
  (values (tied-logits hidden (gpt2-model-wte model))
          (kv-cache (reverse layers) (+ start n))))

(define (last-row logits)
  (define dims (tensor-dims logits))
  (define rows (car dims))
  (define cols (cadr dims))
  (define src (tensor-data logits))
  (define out (make-flvector cols))
  (define base (* (sub1 rows) cols))
  (for ([j (in-range cols)])
    (flvector-set! out j (flvector-ref src (+ base j))))
  out)

(define (next-token-logits model ids)
  (when (null? ids)
    (raise-gpt2 "next-token-logits needs at least one token"))
  (define-values (logits _cache) (model-forward model ids (empty-kv-cache model)))
  (last-row logits))

(define (argmax logits)
  (define n (flvector-length logits))
  (let loop ([i 1] [best 0] [bestv (flvector-ref logits 0)])
    (if (= i n)
        best
        (let ([v (flvector-ref logits i)])
          (if (fl> v bestv)
              (loop (add1 i) i v)
              (loop (add1 i) best bestv))))))

(define (mask-top-k logits k)
  (define n (flvector-length logits))
  (unless (and (exact-positive-integer? k) (<= k n))
    (raise-gpt2
     (format "top-k must be from 1 through the vocabulary size ~a" n)))
  (define order
    (sort (range n)
          (lambda (i j)
            (fl> (flvector-ref logits i) (flvector-ref logits j)))))
  (define out (make-flvector n -inf.0))
  (for ([i (in-list (take order k))])
    (flvector-set! out i (flvector-ref logits i)))
  out)

(define (sample-index probs)
  (define n (flvector-length probs))
  (define u (real->double-flonum (random)))
  (let loop ([i 0] [acc 0.0])
    (cond
      [(= i (sub1 n)) i]
      [else
       (define next (fl+ acc (flvector-ref probs i)))
       (if (fl<= u next)
           i
           (loop (add1 i) next))])))

(define (choose-token logits temperature top-k)
  (cond
    [(not temperature) (argmax logits)]
    [else
     (define n (flvector-length logits))
     (define scaled (make-flvector n))
     (for ([i (in-range n)])
       (flvector-set! scaled i (fl/ (flvector-ref logits i) (real->double-flonum temperature))))
     (define masked (if top-k (mask-top-k scaled top-k) scaled))
     (sample-index (tensor-data (softmax (make-tensor (list n) masked))))]))

(define (emit-text model ids out)
  (define tok (gpt2-model-tokenizer model))
  (cond
    [(not tok) ""]
    [else
     (define dec (make-token-decoder tok))
     (define pieces
       (for/list ([id (in-list ids)]
                  #:unless (= id (gpt2-model-eos-id model)))
         (define piece (token-decoder-add dec id))
         (when out (display piece out))
         piece))
     (define tail (token-decoder-finish dec))
     (when (and out (not (string=? tail "")))
       (display tail out))
     (apply string-append (append pieces (list tail)))]))

(define (model-generate model ids max-new
                        #:temperature [temperature #f]
                        #:top-k [top-k #f]
                        #:seed [seed #f]
                        #:output-port [out #f])
  (unless (exact-positive-integer? max-new)
    (raise-gpt2 "max-new-tokens must be an integer >= 1"))
  (when (null? ids)
    (raise-gpt2 "prompt must encode to at least one token"))
  (define limit (context-limit model))
  (define total (+ (length ids) max-new))
  (when (> total limit)
    (raise-gpt2
     (format "prompt plus generation is longer than the context: ~a tokens, limit ~a"
             total limit)))
  (when (and top-k (> top-k (gpt2-model-vocab-size model)))
    (raise-gpt2
     (format "top-k is larger than the vocabulary: ~a > ~a"
             top-k (gpt2-model-vocab-size model))))
  (when (and seed (>= seed (expt 2 31)))
    (raise-gpt2 "seed must be an integer from 0 through 2147483647"))
  (define gen (and seed (make-pseudo-random-generator)))
  (when seed
    (parameterize ([current-pseudo-random-generator gen])
      (random-seed seed)))
  (define (choose logits)
    (if gen
        (parameterize ([current-pseudo-random-generator gen])
          (choose-token logits temperature top-k))
        (choose-token logits temperature top-k)))
  (define eos (gpt2-model-eos-id model))
  (define-values (first-logits cache)
    (model-forward model ids (empty-kv-cache model)))
  (define first (choose (last-row first-logits)))
  (define new
    (let loop ([cache cache] [chosen first] [acc (list first)] [left (sub1 max-new)])
      (if (or (zero? left) (= chosen eos))
          (reverse acc)
          (let-values ([(logits cache) (model-forward model (list chosen) cache)])
            (define nxt (choose (last-row logits)))
            (loop cache nxt (cons nxt acc) (sub1 left))))))
  (generation new (emit-text model new out)))

(define (config-int cfg key default)
  (define value (hash-ref cfg key default))
  (cond
    [(or (eq? value 'null) (not value)) default]
    [(exact-integer? value) value]
    [(and (real? value) (integer? value)) (inexact->exact value)]
    [else (raise-gpt2 (format "config ~a is not an integer" key))]))

(define (tensor-named tensors name)
  (hash-ref tensors name
            (lambda () (raise-gpt2 (format "missing tensor: ~a" name)))))

(define layer-suffixes
  '("ln_1.weight" "ln_1.bias"
    "attn.c_attn.weight" "attn.c_attn.bias"
    "attn.c_proj.weight" "attn.c_proj.bias"
    "ln_2.weight" "ln_2.bias"
    "mlp.c_fc.weight" "mlp.c_fc.bias"
    "mlp.c_proj.weight" "mlp.c_proj.bias"))

(define (load-model model-dir)
  (define problem (missing-model-assets-message model-dir))
  (when problem
    (raise-gpt2 problem))
  (define cfg
    (with-handlers ([exn:fail:gpt2? (lambda (e) (raise e))]
                    [exn:fail? (lambda (e)
                                 (raise-gpt2 (format "could not read config.json: ~a"
                                                     (exn-message e))))])
      (call-with-input-file (build-path model-dir "config.json") read-json)))
  (unless (hash? cfg)
    (raise-gpt2 "config.json is not an object"))
  (define n-layer (config-int cfg 'n_layer #f))
  (define n-embd (config-int cfg 'n_embd #f))
  (define n-head (config-int cfg 'n_head #f))
  (define vocab (config-int cfg 'vocab_size #f))
  (unless (and n-layer n-embd n-head vocab)
    (raise-gpt2 "config.json is missing n_layer, n_embd, n_head, or vocab_size"))
  (define tok (load-tokenizer (build-path model-dir "vocab.json")
                              (build-path model-dir "merges.txt")))
  (define weights (read-safetensors (build-path model-dir "model.safetensors")))
  (define blocks
    (for/list ([i (in-range n-layer)])
      (define prefix (format "transformer.h.~a." i))
      (define tensors
        (for/hash ([suffix (in-list layer-suffixes)])
          (values suffix (tensor-named weights (string-append prefix suffix)))))
      (block-params-from-tensors #:n-head n-head tensors)))
  (define model
    (make-gpt2 #:wte (tensor-named weights "transformer.wte.weight")
               #:wpe (tensor-named weights "transformer.wpe.weight")
               #:blocks blocks
               #:ln-gain (tensor-named weights "transformer.ln_f.weight")
               #:ln-bias (tensor-named weights "transformer.ln_f.bias")
               #:eos-id (config-int cfg 'eos_token_id endoftext-id)
               #:tokenizer tok))
  (unless (= (gpt2-model-vocab-size model) vocab)
    (raise-gpt2
     (format "wte rows ~a do not match vocab_size ~a"
             (gpt2-model-vocab-size model) vocab)))
  model)
