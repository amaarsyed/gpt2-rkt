#lang racket/base

(module+ test
  ;; `raco test` prints non-void results from a test submodule, so a case
  ;; whose last expression returns an outcome ends with void.
  (require rackunit
           racket/file
           racket/string
           "../main.rkt")

  (struct outcome (code out err) #:transparent)

  (define (run argv)
    (define out (open-output-string))
    (define err (open-output-string))
    (define code (run-cli argv #:output-port out #:error-port err))
    (outcome code (get-output-string out) (get-output-string err)))

  (define (check-help argv pattern)
    (define result (run argv))
    (check-equal? (outcome-code result) 0)
    (check-equal? (outcome-err result) "")
    (check-regexp-match pattern (outcome-out result))
    result)

  (define (check-usage argv pattern)
    (define result (run argv))
    (check-equal? (outcome-code result) 1)
    (check-equal? (outcome-out result) "")
    (check-regexp-match pattern (outcome-err result))
    result)

  (define (generate-argv model-dir . extra)
    (append (list "generate" "--model-dir" model-dir "--prompt" "Once upon a time")
            extra))

  (test-case "top-level help"
    (check-help '("--help") #px"generate")
    (check-help '("-h") #px"Usage:")
    (check-help '("help") #px"generate")
    (void))

  (test-case "no arguments is a usage error that shows help"
    (check-usage '() #px"missing command")
    (check-usage '() #px"generate")
    (void))

  (test-case "generate help lists sampling flags"
    (define result (check-help '("generate" "--help") #px"--max-new-tokens"))
    (for ([flag '("--model-dir" "--prompt" "--temperature" "--top-k" "--seed")])
      (check-regexp-match (regexp-quote flag) (outcome-out result)))
    (check-help '("generate" "-h") #px"greedy")
    (void))

  (test-case "help wins over other generate arguments"
    (check-help '("generate" "--model-dir" "models/gpt2" "--help") #px"--temperature")
    (void))

  (test-case "help flags in value position are values"
    (define dash-h
      (parse-generate-args '("--model-dir" "models/gpt2"
                             "--prompt" "-h"
                             "--max-new-tokens" "4")))
    (check-equal? (generate-config-prompt dash-h) "-h")
    (define dash-dash
      (parse-generate-args '("--model-dir" "m"
                             "--prompt" "--not-a-flag"
                             "--max-new-tokens" "4")))
    (check-equal? (generate-config-prompt dash-dash) "--not-a-flag")
    (check-help '("generate" "--prompt" "-h" "--help") #px"--temperature")
    (check-usage '("generate" "--model-dir" "m" "--prompt" "hi" "--max-new-tokens" "--help")
                 #px"invalid --max-new-tokens: --help")
    (void))

  (test-case "unknown command"
    (check-usage '("nope") #px"unknown command: nope")
    (void))

  (test-case "missing required arguments"
    (check-usage '("generate")
                 #px"missing required arguments: --model-dir, --prompt, --max-new-tokens")
    (void))

  (test-case "flag validation"
    (check-usage '("generate" "--model-dir" "m" "--prompt" "" "--max-new-tokens" "1")
                 #px"prompt must be non-empty")
    (check-usage '("generate" "--model-dir" "m" "--prompt" "hi" "--max-new-tokens" "0")
                 #px"max-new-tokens must be an integer >= 1")
    (check-usage '("generate" "--model-dir" "m" "--prompt" "hi" "--max-new-tokens" "-3")
                 #px"max-new-tokens must be an integer >= 1")
    (check-usage '("generate" "--model-dir" "m" "--prompt" "hi" "--max-new-tokens" "1.5")
                 #px"max-new-tokens must be an integer >= 1")
    (check-usage '("generate" "--model-dir" "m" "--prompt" "hi" "--max-new-tokens" "nope")
                 #px"invalid --max-new-tokens: nope")
    (check-usage '("generate" "--model-dir" "m" "--prompt" "hi" "--max-new-tokens" "1"
                               "--temperature" "0")
                 #px"temperature must be a finite number > 0")
    (check-usage '("generate" "--model-dir" "m" "--prompt" "hi" "--max-new-tokens" "1"
                               "--temperature" "-0.1")
                 #px"temperature must be a finite number > 0")
    (check-usage '("generate" "--model-dir" "m" "--prompt" "hi" "--max-new-tokens" "1"
                               "--temperature" "hot")
                 #px"invalid --temperature: hot")
    (check-usage '("generate" "--model-dir" "m" "--prompt" "hi" "--max-new-tokens" "1"
                               "--temperature" "+inf.0")
                 #px"temperature must be a finite number > 0")
    (check-usage '("generate" "--model-dir" "m" "--prompt" "hi" "--max-new-tokens" "1"
                               "--temperature" "+nan.0")
                 #px"temperature must be a finite number > 0")
    (check-usage '("generate" "--model-dir" "m" "--prompt" "hi" "--max-new-tokens" "1"
                               "--temperature" "0.8" "--top-k" "0")
                 #px"top-k must be an integer >= 1")
    (check-usage '("generate" "--model-dir" "m" "--prompt" "hi" "--max-new-tokens" "1"
                               "--temperature" "0.8" "--seed" "-1")
                 #px"seed must be an integer >= 0")
    (check-usage '("generate" "--model-dir" "m" "--prompt" "hi" "--max-new-tokens" "1"
                               "--top-k" "40")
                 #px"top-k requires temperature > 0")
    (check-usage '("generate" "--model-dir" "m" "--prompt" "hi" "--max-new-tokens" "1"
                               "--seed" "1")
                 #px"seed requires temperature > 0")
    (check-usage '("generate" "--model-dir" "m" "--prompt" "hi" "--max-new-tokens" "1"
                               "--unknown" "x")
                 #px"unknown flag: --unknown")
    (check-usage '("generate" "--model-dir" "m" "--prompt" "hi" "--max-new-tokens")
                 #px"missing value for --max-new-tokens")
    (check-usage '("generate" "--prompt" "hi" "extra")
                 #px"unexpected argument: extra")
    (check-usage '("generate" "--model-dir" "m" "--model-dir" "n" "--prompt" "hi"
                               "--max-new-tokens" "1")
                 #px"duplicate flag: --model-dir")
    (void))

  (test-case "greedy is the default and sampling flags parse"
    (define greedy
      (parse-generate-args '("--model-dir" "models/gpt2"
                             "--prompt" "Once upon a time"
                             "--max-new-tokens" "16")))
    (check-equal? (generate-config-model-dir greedy) "models/gpt2")
    (check-equal? (generate-config-prompt greedy) "Once upon a time")
    (check-equal? (generate-config-max-new-tokens greedy) 16)
    (check-equal? (generate-config-temperature greedy) #f)
    (check-equal? (generate-config-top-k greedy) #f)
    (check-equal? (generate-config-seed greedy) #f))

  (test-case "sampling config stores temperature, top-k, and seed"
    (define cfg
      (parse-generate-args '("--temperature" "0.8"
                             "--seed" "1"
                             "--max-new-tokens" "16"
                             "--top-k" "40"
                             "--prompt" "Once upon a time"
                             "--model-dir" "models/gpt2")))
    (check-equal? (generate-config-temperature cfg) 0.8)
    (check-equal? (generate-config-top-k cfg) 40)
    (check-equal? (generate-config-seed cfg) 1)
    (check-exn exn:fail:gpt2?
               (lambda ()
                 (parse-generate-args '("--model-dir" "models/gpt2"
                                        "--prompt" "hi"
                                        "--max-new-tokens=4"
                                        "--temperature=0")))))

  (test-case "constructor rejects an invalid config"
    (check-exn exn:fail:gpt2?
               (lambda () (generate-config "dir" "prompt" 0 #f #f #f)))
    (check-exn exn:fail:contract?
               (lambda () (generate "not a config"))))

  (test-case "missing model directory"
    (define argv
      #("generate" "--model-dir" "models/gpt2"
        "--prompt" "Once upon a time" "--max-new-tokens" "16"))
    (define result (check-usage argv #px"missing model assets"))
    (check-regexp-match #px"directory does not exist" (outcome-err result))
    (check-regexp-match #px"config.json" (outcome-err result))
    (check-regexp-match #px"model.safetensors" (outcome-err result))
    (check-exn
     (lambda (e)
       (and (exn:fail:gpt2? e)
            (string-contains? (exn-message e) "missing model assets")))
     (lambda ()
       (generate (generate-config "models/gpt2" "Once upon a time" 16 #f #f #f)))))

  (test-case "model path that is a file"
    (define path (make-temporary-file "gpt2-rkt-~a"))
    (dynamic-wind
     void
     (lambda ()
       (check-usage (generate-argv (path->string path) "--max-new-tokens" "1")
                    #px"not a directory")
       (void))
     (lambda () (delete-file path))))

  (test-case "partial model directory names the missing files"
    (define dir (make-temporary-file "gpt2-rkt-~a" 'directory))
    (dynamic-wind
     void
     (lambda ()
       (call-with-output-file (build-path dir "config.json")
         void
         #:exists 'truncate)
       (define result
         (check-usage (generate-argv (path->string dir) "--max-new-tokens" "8")
                      #px"missing model assets"))
       (check-regexp-match #px"vocab.json" (outcome-err result))
       (check-regexp-match #px"merges.txt" (outcome-err result))
       (check-regexp-match #px"model.safetensors" (outcome-err result))
       (check-false (string-contains? (outcome-err result) "config.json")))
     (lambda () (delete-directory/files dir))))

  (test-case "empty assets fail while reading the config"
    (define dir (make-temporary-file "gpt2-rkt-~a" 'directory))
    (dynamic-wind
     void
     (lambda ()
       (for ([name (in-list required-model-asset-names)])
         (call-with-output-file (build-path dir name) void #:exists 'truncate))
       (define result
         (check-usage (generate-argv (path->string dir) "--max-new-tokens" "2"
                                     "--temperature" "0.5" "--top-k" "4" "--seed" "7")
                      #px"config.json"))
       (check-false (regexp-match? #px"missing model assets" (outcome-err result))))
     (lambda () (delete-directory/files dir))))

  (test-case "run-cli rejects a non-string argument"
    (check-exn exn:fail:contract?
               (lambda () (run-cli '("generate" 1))))))
