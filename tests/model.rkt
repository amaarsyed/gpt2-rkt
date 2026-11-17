#lang racket/base

(module+ test
  (require racket/flonum
           racket/list
           rackunit
           "../src/block.rkt"
           "../src/errors.rkt"
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

  (test-case "generation stops at the eos token"
    (define wte (make-tensor '(4 2) (flvector 0.0 1.0
                                              0.0 1.0
                                              0.0 1.0
                                              2.0 0.0)))
    (define model
      (make-gpt2 #:wte wte
                 #:wpe (filled '(4 2) 0.0)
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
  )