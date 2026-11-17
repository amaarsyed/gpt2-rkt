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
       (check-exn #rx"F16"
                  (lambda () (read-safetensors path)))
       (check-exn #rx"row"
                  (lambda () (read-safetensors path))))
     (lambda () (delete-file path))))

  (test-case "an unaligned header and a space-padded header both read"
    (define body (real->floating-point-bytes 3.25 4 #f))
    (define header
      (let loop ([raw (string->bytes/utf-8
                       "{\"row\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}")])
        (if (= (remainder (bytes-length raw) 8) 3)
            raw
            (loop (bytes-append raw #" ")))))
    (check-equal? (remainder (bytes-length header) 8) 3)
    (check-not-equal? (remainder (+ 8 (bytes-length header)) 4) 0)
    (define odd (write-file header body))
    (define padded
      (write-file (bytes-append header (make-bytes (- 8 (remainder (bytes-length header) 8)) 32))
                  body))
    (dynamic-wind
     void
     (lambda ()
       (check-equal? (elements (hash-ref (read-safetensors odd) "row")) '(3.25))
       (check-equal? (elements (hash-ref (read-safetensors padded) "row")) '(3.25)))
     (lambda ()
       (delete-file odd)
       (delete-file padded))))

  (test-case "a byte length that disagrees with the shape is a gpt2 error"
    (define header
      (string->bytes/utf-8
       "{\"row\":{\"dtype\":\"F32\",\"shape\":[1,2],\"data_offsets\":[0,4]}}"))
    (define path (write-file header (real->floating-point-bytes 1.0 4 #f)))
    (dynamic-wind
     void
     (lambda ()
       (check-exn exn:fail:gpt2?
                  (lambda () (read-safetensors path)))
       (check-exn #rx"row"
                  (lambda () (read-safetensors path))))
     (lambda () (delete-file path))))

  (test-case "a skipped attn.bias buffer is not decoded"
    (define header
      (string->bytes/utf-8
       (string-append
        "{\"h.0.attn.bias\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,2]},"
        "\"wte.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[2,6]}}")))
    (define body
      (bytes-append #"ab" (real->floating-point-bytes 4.5 4 #f)))
    (define path (write-file header body))
    (dynamic-wind
     void
     (lambda ()
       (define tensors
         (read-safetensors path #:skip (lambda (name) (regexp-match? #rx"attn\\.bias$" name))))
       (check-false (hash-has-key? tensors "h.0.attn.bias"))
       (check-equal? (elements (hash-ref tensors "wte.weight")) '(4.5)))
     (lambda () (delete-file path))))

  (test-case "an empty shape is a gpt2 error that names the tensor"
    (define header
      #"{\"scalar\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[0,4]}}")
    (define path (write-file header (real->floating-point-bytes 1.0 4 #f)))
    (dynamic-wind
     void
     (lambda ()
       (check-exn
        (lambda (e)
          (and (exn:fail:gpt2? e)
               (regexp-match? #rx"shape is invalid: scalar" (exn-message e))))
        (lambda () (read-safetensors path))))
     (lambda () (delete-file path))))
  )
