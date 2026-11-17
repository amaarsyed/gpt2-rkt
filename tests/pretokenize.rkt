#lang racket/base

(module+ test
  (require rackunit
           "../src/pretokenize.rkt")

  (test-case "contractions are case-sensitive"
    (check-equal? (gpt2-pretokenize "I'm") '("I" "'m"))
    (check-equal? (gpt2-pretokenize "I'M") '("I" "'" "M"))
    (check-equal? (gpt2-pretokenize "don't") '("don" "'t"))
    (check-equal? (gpt2-pretokenize "DON'T") '("DON" "'" "T"))
    (check-equal? (gpt2-pretokenize "'s") '("'s"))
    (check-equal? (gpt2-pretokenize "'S") '("'" "S")))

  (test-case "trailing and repeated spaces split like GPT-2"
    (check-equal? (gpt2-pretokenize "  ") '("  "))
    (check-equal? (gpt2-pretokenize "hello ") '("hello" " "))
    (check-equal? (gpt2-pretokenize "  a") '(" " " a"))
    (check-equal? (gpt2-pretokenize "a  b  ") '("a" " " " b" "  "))
    (check-equal? (gpt2-pretokenize "hello  world") '("hello" " " " world"))
    (check-equal? (gpt2-pretokenize "\n\n") '("\n\n")))

  (test-case "Unicode whitespace is not treated as punctuation"
    (check-equal? (gpt2-pretokenize "\u00A0") '("\u00A0"))
    (check-equal? (gpt2-pretokenize "\u3000") '("\u3000"))
    (check-equal? (gpt2-pretokenize "\u00A0hello") '("\u00A0" "hello"))
    ;; An ASCII-only whitespace class splits this into a newline and a NBSP.
    (check-equal? (gpt2-pretokenize "\n\u00A0") '("\n\u00A0"))
    (check-equal? (gpt2-pretokenize "a\n\u00A0") '("a" "\n\u00A0")))

  (test-case "letters, digits, and punctuation stay in their own runs"
    (check-equal? (gpt2-pretokenize "") '())
    (check-equal? (gpt2-pretokenize "café") '("café"))
    (check-equal? (gpt2-pretokenize "你好") '("你好"))
    (check-equal? (gpt2-pretokenize "12345") '("12345"))
    (check-equal? (gpt2-pretokenize "...") '("..."))
    (check-equal? (gpt2-pretokenize "GPT-2") '("GPT" "-" "2")))
  )