#lang racket/base

(module+ test
  (require json
           racket/file
           racket/runtime-path
           racket/string
           racket/system
           rackunit
           "../src/merges.rkt"
           "../src/tokenizer.rkt")

  (define-runtime-path vocab-path "../models/gpt2/vocab.json")
  (define-runtime-path merges-path "../models/gpt2/merges.txt")
  (define-runtime-path cases-path "tokenizer-cases.json")

  (unless (and (file-exists? vocab-path) (file-exists? merges-path))
    (error 'tokenizer-cases
           "missing BPE files under models/gpt2; run tools/fetch-gpt2-vocab.sh"))

  (define (sha256-file path)
    (define out (open-output-string))
    (parameterize ([current-output-port out])
      (unless (system* (find-executable-path "sha256sum") (path->string path))
        (error 'tokenizer-cases "sha256sum failed for ~a" path)))
    (car (string-split (get-output-string out))))

  (define vocab-sha256
    "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783")
  (define merges-sha256
    "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5")

  (test-case "pinned BPE files match the published digests"
    (check-equal? (sha256-file vocab-path) vocab-sha256)
    (check-equal? (sha256-file merges-path) merges-sha256))

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