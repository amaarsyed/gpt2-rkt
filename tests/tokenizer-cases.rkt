#lang racket/base

(module+ test
  (require file/sha1
           json
           racket/file
           racket/runtime-path
           racket/string
           rackunit
           "../src/merges.rkt"
           "../src/tokenizer.rkt")

  (define-runtime-path vocab-path "../models/gpt2/vocab.json")
  (define-runtime-path merges-path "../models/gpt2/merges.txt")
  (define-runtime-path assets-path "../tools/gpt2-assets.sha256")
  (define-runtime-path cases-path "tokenizer-cases.json")

  (unless (and (file-exists? vocab-path) (file-exists? merges-path))
    (error 'tokenizer-cases
           "missing BPE files under models/gpt2; run tools/fetch-gpt2-vocab.sh"))

  (define (sha256-file path)
    (bytes->hex-string (call-with-input-file path sha256-bytes)))

  (define (read-asset-digests path)
    (call-with-input-file path
      (lambda (in)
        (for/hash ([line (in-lines in)]
                   #:unless (string=? (string-trim line) ""))
          (define parts (string-split line))
          (unless (= (length parts) 2)
            (error 'tokenizer-cases "malformed digest line: ~a" line))
          (values (cadr parts) (string-downcase (car parts)))))))

  (define digests (read-asset-digests assets-path))

  (test-case "digest lines may be blank"
    (define path (make-temporary-file "gpt2-digests-~a"))
    (dynamic-wind
     void
     (lambda ()
       (call-with-output-file path
         (lambda (out)
           (displayln "" out)
           (displayln "abc vocab.json" out)
           (displayln "   " out)
           (displayln "def merges.txt" out))
         #:exists 'truncate)
       (define digests (read-asset-digests path))
       (check-equal? (hash-ref digests "vocab.json") "abc")
       (check-equal? (hash-ref digests "merges.txt") "def"))
     (lambda () (delete-file path))))

  (test-case "pinned BPE files match the published digests"
    (check-equal? (sha256-file vocab-path) (hash-ref digests "vocab.json"))
    (check-equal? (sha256-file merges-path) (hash-ref digests "merges.txt")))

  (test-case "the published merges file still starts with #version"
    (define text (bytes->string/utf-8 (file->bytes merges-path)))
    (check-true (string-prefix? text "#version:"))
    (define ranks (read-merge-ranks text))
    (check-equal? (hash-ref ranks (cons "\u0120" "t")) 0)
    (check-false (hash-has-key? ranks (cons "#version:" "0.2"))))

  (define tok (load-tokenizer vocab-path merges-path))
  (define cases
    (hash-ref (call-with-input-file cases-path read-json) 'cases))

  (test-case "encode matches the reference ids and decode round-trips"
    (for ([c (in-list cases)])
      (define text (hash-ref c 'text))
      (define ids (hash-ref c 'ids))
      (define got (encode tok text))
      (check-equal? got ids)
      (check-equal? (decode tok got) text)
      (check-equal? (decode tok ids) text))
    (check-equal? (encode tok "<|endoftext|>") (list endoftext-id)))
  )