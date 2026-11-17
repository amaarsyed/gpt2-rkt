#lang racket/base

;; Expected files in a `--model-dir`. Names follow the Hugging Face GPT-2
;; layout this engine will load: config, BPE vocab and merges, safetensors.

(require racket/string)

(provide required-model-asset-names
         missing-model-assets-message)

(define required-model-asset-names
  '("config.json"
    "vocab.json"
    "merges.txt"
    "model.safetensors"))

(define expected-files-line
  (string-append "expected: " (string-join required-model-asset-names ", ")))

;; #f when every required file is present. Otherwise a user-facing message
;; that begins with "missing model assets".
(define (missing-model-assets-message model-dir)
  (cond
    [(file-exists? model-dir)
     (format "missing model assets: ~a (not a directory)\n~a"
             model-dir
             expected-files-line)]
    [(not (directory-exists? model-dir))
     (format "missing model assets: ~a (directory does not exist)\n~a"
             model-dir
             expected-files-line)]
    [else
     (define missing
       (filter (lambda (name)
                 (not (file-exists? (build-path model-dir name))))
               required-model-asset-names))
     (if (null? missing)
         #f
         (format "missing model assets: ~a\nmissing files: ~a"
                 model-dir
                 (string-join missing ", ")))]))
