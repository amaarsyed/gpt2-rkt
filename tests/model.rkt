#lang racket/base

(module+ test
  (require json
           racket/file
           racket/flonum
           racket/list
           racket/string
           rackunit
           "../src/block.rkt"
           "../src/errors.rkt"
           "../src/model-assets.rkt"
           "../src/model.rkt"
           "../src/nn.rkt"
           "../src/tensor.rkt")

  (define (elements data)
    (for/list ([i (in-range (flvector-length data))])
      (flvector-ref data i)))

  (define (filled dims value)
    (define n (apply * dims))
    (make-tensor dims (make-flvector n value)))

  (define (numbered rows cols)
    (define data (make-flvector (* rows cols)))
    (for ([i (in-range rows)])
      (for ([j (in-range cols)])
        (flvector-set! data (+ (* i cols) j)
                       (exact->inexact (remainder (+ (* i 17) (* j j)) 11)))))
    (make-tensor (list rows cols) data))

  (define (block d n-head salt)
    (define inner (* 4 d))
    (make-block-params #:n-head n-head
                       #:ln1-gain (filled (list d) 1.0)
                       #:ln1-bias (filled (list d) 0.0)
                       #:c-attn-w (numbered d (* 3 d))
                       #:c-attn-b (filled (list (* 3 d)) 0.0)
                       #:attn-proj-w (numbered d d)
                       #:attn-proj-b (filled (list d) 0.0)
                       #:ln2-gain (filled (list d) 1.0)
                       #:ln2-bias (filled (list d) 0.0)
                       #:c-fc-w (numbered d inner)
                       #:c-fc-b (filled (list inner) 0.0)
                       #:mlp-proj-w (numbered inner d)
                       #:mlp-proj-b (filled (list d) 0.0)))

  (define (zero-block d)
    (define inner (* 4 d))
    (make-block-params #:n-head 1
                       #:ln1-gain (filled (list d) 0.0)
                       #:ln1-bias (filled (list d) 0.0)
                       #:c-attn-w (filled (list d (* 3 d)) 0.0)
                       #:c-attn-b (filled (list (* 3 d)) 0.0)
                       #:attn-proj-w (filled (list d d) 0.0)
                       #:attn-proj-b (filled (list d) 0.0)
                       #:ln2-gain (filled (list d) 0.0)
                       #:ln2-bias (filled (list d) 0.0)
                       #:c-fc-w (filled (list d inner) 0.0)
                       #:c-fc-b (filled (list inner) 0.0)
                       #:mlp-proj-w (filled (list inner d) 0.0)
                       #:mlp-proj-b (filled (list d) 0.0)))

  (test-case "tied logits are h dotted with each wte row"
    (define wte (make-tensor '(3 2) (flvector 1.0 0.0
                                              0.0 1.0
                                              1.0 1.0)))
    (define model
      (make-gpt2 #:wte wte
                 #:wpe (filled '(4 2) 0.0)
                 #:blocks (list (zero-block 2))
                 #:ln-gain (filled '(2) 0.0)
                 #:ln-bias (make-tensor '(2) (flvector 1.0 0.0))
                 #:eos-id 0))
    (check-equal? (elements (next-token-logits model '(1))) '(1.0 0.0 1.0))
    (check-exn #rx"columns"
               (lambda ()
                 (matmul wte (make-tensor '(1 2) (flvector 1.0 0.0))
                         (filled '(2) 0.0)))))

  (test-case "cached logits match a full recompute at every step"
    (define model
      (make-gpt2 #:wte (numbered 5 4)
                 #:wpe (numbered 8 4)
                 #:blocks (list (block 4 2 1) (block 4 2 2))
                 #:ln-gain (filled '(4) 1.0)
                 #:ln-bias (filled '(4) 0.0)
                 #:eos-id 0))
    (define ids '(1 3 2 4))
    (define full
      (for/list ([t (in-range 1 (add1 (length ids)))])
        (elements (next-token-logits model (take ids t)))))
    (define incremental
      (let loop ([cache (empty-kv-cache model)] [seen 0] [got '()])
        (if (= seen (length ids))
            (reverse got)
            (let-values ([(logits cache)
                          (model-forward model (list (list-ref ids seen)) cache)])
              (loop cache
                    (add1 seen)
                    (cons (elements (tensor-data logits)) got))))))
    (check-equal? incremental full)
    (check-false (equal? (car full) (cadr full))))

  (test-case "generation stops on token 50256"
    (define vocab 50257)
    (define data (make-flvector (* vocab 2) 0.0))
    (flvector-set! data (* 50256 2) 1.0)
    (define model
      (make-gpt2 #:wte (make-tensor (list vocab 2) data)
                 #:wpe (filled '(8 2) 0.0)
                 #:blocks (list (zero-block 2))
                 #:ln-gain (filled '(2) 0.0)
                 #:ln-bias (make-tensor '(2) (flvector 1.0 0.0))
                 #:eos-id 50256))
    (define result (model-generate model '(0) 4))
    (check-equal? (generation-token-ids result) '(50256))
    (check-equal? (generation-text result) ""))

  (test-case "generation stops at the eos token"
    (define wte (make-tensor '(4 2) (flvector 0.0 1.0
                                              0.0 1.0
                                              0.0 1.0
                                              2.0 0.0)))
    (define model
      (make-gpt2 #:wte wte
                 #:wpe (filled '(8 2) 0.0)
                 #:blocks (list (zero-block 2))
                 #:ln-gain (filled '(2) 0.0)
                 #:ln-bias (make-tensor '(2) (flvector 1.0 0.0))
                 #:eos-id 3))
    (define result (model-generate model '(0) 5))
    (check-equal? (generation-token-ids result) '(3)))

  (test-case "exactly 1024 positions are legal and 1025 is not"
    (define model
      (make-gpt2 #:wte (filled '(2 2) 0.0)
                 #:wpe (filled '(1024 2) 0.0)
                 #:blocks (list (zero-block 2))
                 #:ln-gain (filled '(2) 0.0)
                 #:ln-bias (filled '(2) 0.0)
                 #:eos-id 1))
    (check-equal? (flvector-length (next-token-logits model (make-list 1024 0))) 2)
    (check-exn #rx"1024"
               (lambda () (next-token-logits model (make-list 1025 0))))
    (check-equal? (generation-token-ids (model-generate model (make-list 1023 0) 1))
                  '(0))
    (check-exn #rx"1024"
               (lambda () (model-generate model (make-list 1024 0) 1))))

  (test-case "a seed repeats the sample and top-k 1 is greedy"
    (define wte (make-tensor '(4 2) (flvector 1.0 0.0
                                              0.9 0.0
                                              0.8 0.0
                                              0.7 0.0)))
    (define model
      (make-gpt2 #:wte wte
                 #:wpe (filled '(16 2) 0.0)
                 #:blocks (list (zero-block 2))
                 #:ln-gain (filled '(2) 0.0)
                 #:ln-bias (make-tensor '(2) (flvector 1.0 0.0))
                 #:eos-id 9))
    (define a (generation-token-ids
               (model-generate model '(0) 4 #:temperature 1.0 #:seed 1)))
    (define b (generation-token-ids
               (model-generate model '(0) 4 #:temperature 1.0 #:seed 1)))
    (define c (generation-token-ids
               (model-generate model '(0) 4 #:temperature 1.0 #:seed 2)))
    (check-equal? a b)
    (check-false (equal? a c))
    (check-equal? (generation-token-ids
                   (model-generate model '(0) 3 #:temperature 0.5 #:top-k 1 #:seed 1))
                  '(0 0 0))
    (check-exn #rx"vocabulary"
               (lambda ()
                 (model-generate model '(0) 1 #:temperature 0.5 #:top-k 5 #:seed 1))))

  (test-case "generation uses the model's position count"
    (define model
      (make-gpt2 #:wte (filled '(2 2) 0.0)
                 #:wpe (filled '(32 2) 0.0)
                 #:blocks (list (zero-block 2))
                 #:ln-gain (filled '(2) 0.0)
                 #:ln-bias (filled '(2) 0.0)
                 #:eos-id 1))
    (check-equal? (gpt2-model-n-positions model) 32)
    (check-exn exn:fail:gpt2?
               (lambda () (model-generate model '(0) 40)))
    (check-exn #rx"limit 32"
               (lambda () (model-generate model '(0) 40)))
    (check-exn #rx"limit 32"
               (lambda () (next-token-logits model (make-list 33 0))))
    (check-equal? (length (generation-token-ids (model-generate model (make-list 31 0) 1)))
                  1))

  (define (json-ints nums)
    (string-append "[" (string-join (map number->string nums) ",") "]"))

  (define (write-zero-safetensors path specs)
    (define entries '())
    (define blobs '())
    (define offset 0)
    (for ([spec (in-list specs)])
      (define dims (cdr spec))
      (define nbytes (* 4 (apply * dims)))
      (set! entries
            (cons (format "~s:{\"dtype\":\"F32\",\"shape\":~a,\"data_offsets\":~a}"
                          (car spec)
                          (json-ints dims)
                          (json-ints (list offset (+ offset nbytes))))
                  entries))
      (set! blobs (cons (make-bytes nbytes 0) blobs))
      (set! offset (+ offset nbytes)))
    (define payload
      (string->bytes/utf-8
       (string-append "{" (string-join (reverse entries) ",") "}")))
    (define padded
      (bytes-append payload (make-bytes (modulo (- 8 (modulo (bytes-length payload) 8)) 8) 32)))
    (call-with-output-file path
      (lambda (out)
        (write-bytes (integer->integer-bytes (bytes-length padded) 8 #f #f) out)
        (write-bytes padded out)
        (for ([blob (in-list (reverse blobs))])
          (write-bytes blob out)))
      #:exists 'truncate))

  (test-case "a config without eos_token_id stops on 50256"
    (define dir (make-temporary-file "gpt2-rkt-~a" 'directory))
    (dynamic-wind
     void
     (lambda ()
       (call-with-output-file (build-path dir "config.json")
         (lambda (out)
           (write-json (hash 'n_layer 1 'n_embd 2 'n_head 1 'n_positions 4 'vocab_size 2) out))
         #:exists 'truncate)
       (call-with-output-file (build-path dir "vocab.json")
         (lambda (out) (write-json (hash 'a 0 'b 1) out))
         #:exists 'truncate)
       (call-with-output-file (build-path dir "merges.txt")
         (lambda (out) (displayln "#version: 0.2" out))
         #:exists 'truncate)
       (write-zero-safetensors
        (build-path dir "model.safetensors")
        `(("transformer.wte.weight" 2 2)
          ("transformer.wpe.weight" 4 2)
          ("transformer.ln_f.weight" 2)
          ("transformer.ln_f.bias" 2)
          ("transformer.h.0.ln_1.weight" 2)
          ("transformer.h.0.ln_1.bias" 2)
          ("transformer.h.0.attn.c_attn.weight" 2 6)
          ("transformer.h.0.attn.c_attn.bias" 6)
          ("transformer.h.0.attn.c_proj.weight" 2 2)
          ("transformer.h.0.attn.c_proj.bias" 2)
          ("transformer.h.0.ln_2.weight" 2)
          ("transformer.h.0.ln_2.bias" 2)
          ("transformer.h.0.mlp.c_fc.weight" 2 8)
          ("transformer.h.0.mlp.c_fc.bias" 8)
          ("transformer.h.0.mlp.c_proj.weight" 8 2)
          ("transformer.h.0.mlp.c_proj.bias" 2)))
       (define model (load-model dir))
       (check-equal? (gpt2-model-eos-id model) 50256)
       (check-equal? (gpt2-model-n-positions model) 4))
     (lambda () (delete-directory/files dir))))

  (define (write-text path text)
    (call-with-output-file path
      (lambda (out) (display text out))
      #:exists 'truncate))

  (define (tiny-dir specs config)
    (define dir (make-temporary-file "gpt2-rkt-~a" 'directory))
    (write-text (build-path dir "config.json") config)
    (write-json-file (build-path dir "vocab.json") (hash 'a 0 'b 1))
    (write-text (build-path dir "merges.txt") "#version: 0.2\n")
    (write-zero-safetensors (build-path dir "model.safetensors") specs)
    dir)

  (define (write-json-file path jsexpr)
    (call-with-output-file path
      (lambda (out) (write-json jsexpr out))
      #:exists 'truncate))

  (define base-config
    "{\"n_layer\":1,\"n_embd\":2,\"n_head\":1,\"n_positions\":4,\"vocab_size\":2}")

  (define base-specs
    '(("wte.weight" 2 2)
      ("wpe.weight" 4 2)
      ("ln_f.weight" 2)
      ("ln_f.bias" 2)
      ("h.0.ln_1.weight" 2)
      ("h.0.ln_1.bias" 2)
      ("h.0.attn.c_attn.weight" 2 6)
      ("h.0.attn.c_attn.bias" 6)
      ("h.0.attn.c_proj.weight" 2 2)
      ("h.0.attn.c_proj.bias" 2)
      ("h.0.ln_2.weight" 2)
      ("h.0.ln_2.bias" 2)
      ("h.0.mlp.c_fc.weight" 2 8)
      ("h.0.mlp.c_fc.bias" 8)
      ("h.0.mlp.c_proj.weight" 8 2)
      ("h.0.mlp.c_proj.bias" 2)
      ("h.0.attn.bias" 1 1 1 1)))

  (test-case "Hugging Face names load without a transformer prefix"
    (define dir (tiny-dir base-specs base-config))
    (dynamic-wind
     void
     (lambda ()
       (define model (load-model dir))
       (check-equal? (gpt2-model-n-embd model) 2)
       (check-equal? (flvector-length (next-token-logits model '(0))) 2))
     (lambda () (delete-directory/files dir))))

  (test-case "a transposed c_attn, a missing key, and an extra key are named"
    (define transposed
      (map (lambda (spec)
             (if (equal? (car spec) "h.0.attn.c_attn.weight")
                 '("h.0.attn.c_attn.weight" 6 2)
                 spec))
           base-specs))
    (define dir (tiny-dir transposed base-config))
    (dynamic-wind
     void
     (lambda ()
       (check-exn #rx"h.0.attn.c_attn.weight"
                  (lambda () (load-model dir))))
     (lambda () (delete-directory/files dir)))
    (define missing (filter (lambda (spec) (not (equal? (car spec) "wte.weight"))) base-specs))
    (define dir2 (tiny-dir missing base-config))
    (dynamic-wind
     void
     (lambda ()
       (check-exn #rx"missing tensor: wte.weight"
                  (lambda () (load-model dir2))))
     (lambda () (delete-directory/files dir2)))
    (define extra (append base-specs '(("extra.weight" 2))))
    (define dir3 (tiny-dir extra base-config))
    (dynamic-wind
     void
     (lambda ()
       (check-exn #rx"unexpected tensor: extra.weight"
                  (lambda () (load-model dir3))))
     (lambda () (delete-directory/files dir3))))

  (test-case "a sha256 mismatch names the file and the fetch command"
    (define path (make-temporary-file "gpt2-rkt-~a"))
    (dynamic-wind
     void
     (lambda ()
       (call-with-output-file path
         (lambda (out) (display "corrupt" out))
         #:exists 'truncate)
       (check-exn #rx"sha256 mismatch"
                  (lambda () (check-file-sha256 path "0000000000000000000000000000000000000000000000000000000000000000")))
       (check-exn #px"sh tools/fetch-gpt2-vocab.sh"
                  (lambda () (check-file-sha256 path "0000000000000000000000000000000000000000000000000000000000000000"))))
     (lambda () (delete-file path))))
  )