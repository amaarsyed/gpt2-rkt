#lang racket/base

;; Expected files in a `--model-dir`. Names follow the Hugging Face GPT-2
;; layout this engine will load: config, BPE vocab and merges, safetensors.

(require file/sha1
         racket/string
         "errors.rkt")

(provide asset-fetch-command
         required-model-asset-names
         missing-model-assets-message
         check-file-sha256)

(define asset-fetch-command "sh tools/fetch-gpt2-vocab.sh")

(define required-model-asset-names
  '("config.json"
    "vocab.json"
    "merges.txt"
    "model.safetensors"))

(define expected-files-line
  (string-append "expected: " (string-join required-model-asset-names ", ")))

(define (with-fetch-hint message)
  (string-append message "\n" asset-fetch-command))

;; #f when every required file is present. Otherwise a user-facing message
;; that begins with "missing model assets" and names the fetch command.
(define (missing-model-assets-message model-dir)
  (cond
    [(file-exists? model-dir)
     (with-fetch-hint
      (format "missing model assets: ~a (not a directory)\n~a"
              model-dir
              expected-files-line))]
    [(not (directory-exists? model-dir))
     (with-fetch-hint
      (format "missing model assets: ~a (directory does not exist)\n~a"
              model-dir
              expected-files-line))]
    [else
     (define missing
       (filter (lambda (name)
                 (not (file-exists? (build-path model-dir name))))
               required-model-asset-names))
     (if (null? missing)
         #f
         (with-fetch-hint
          (format "missing model assets: ~a\nmissing files: ~a"
                  model-dir
                  (string-join missing ", "))))]))

(define (check-file-sha256 path expected)
  (define got
    (bytes->hex-string
     (call-with-input-file path sha256-bytes #:mode 'binary)))
  (unless (string=? (string-downcase got) (string-downcase expected))
    (raise-gpt2
     (format "sha256 mismatch for ~a\nexpected ~a\ngot ~a\n~a"
             path expected got asset-fetch-command))))
