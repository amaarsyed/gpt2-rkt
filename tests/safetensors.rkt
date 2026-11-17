#lang racket/base

(module+ test
  (require racket/file
           racket/flonum
           rackunit
           "../src/errors.rkt"
           "../src/safetensors.rkt"
           "../src/tensor.rkt")

  (define (elements t)
    (define data (tensor-data t))
    (for/list ([i (in-range (flvector-length data))])
      (flvector-ref data i)))

  (define (write-file header body)
    (define path (make-temporary-file "safetensors-~a"))
    (call-with-output-file path
      (lambda (out)
        (write-bytes (integer->integer-bytes (bytes-length header) 8 #f #f) out)
        (write-bytes header out)
        (write-bytes body out))
      #:exists 'truncate)
    path)

  (test-case "F32 tensors round-trip through the header"
    (define header
      (string->bytes/utf-8
       "{\"row\":{\"dtype\":\"F32\",\"shape\":[1,2],\"data_offsets\":[0,8]},\"__metadata__\":{\"format\":\"pt\"}}"))
    (define body
      (bytes-append (real->floating-point-bytes 1.5 4 #f)
                    (real->floating-point-bytes -2.0 4 #f)))
    (define path (write-file header body))
    (dynamic-wind
     void
     (lambda ()
       (define tensors (read-safetensors path))
       (check-equal? (tensor-dims (hash-ref tensors "row")) '(1 2))
       (check-equal? (elements (hash-ref tensors "row")) '(1.5 -2.0))
       (check-false (hash-has-key? tensors "__metadata__")))
     (lambda () (delete-file path))))

  (test-case "a non-F32 dtype is rejected"
    (define header
      (string->bytes/utf-8
       "{\"row\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,2]}}"))
    (define path (write-file header #"ab"))
    (dynamic-wind
     void
     (lambda ()
       (check-exn exn:fail:gpt2?
                  (lambda () (read-safetensors path))))
     (lambda () (delete-file path))))
  )