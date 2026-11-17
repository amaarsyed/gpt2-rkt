#lang racket/base

;; GPT-2 pre-tokenizer, the regex from encoder.py:
;;   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
;;
;; pregexp's \s is ASCII-only. The class below is the Unicode White_Space
;; set that Python's regex module uses for \s (25 code points, not U+180E).
;; The trailing alternative is \s+(?!\S): whitespace that is not followed
;; by a non-whitespace. That is not the same as (?!\s).

(provide gpt2-pretokenize)

(define ws-chars
  (string-append
   "\u0009\u000A\u000B\u000C\u000D\u0020\u0085\u00A0\u1680"
   "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200A"
   "\u2028\u2029\u202F\u205F\u3000"))

(define gpt2-pretoken-pattern
  (let ([ws (string-append "[" ws-chars "]")]
        [not-ws (string-append "[^" ws-chars "]")])
    (pregexp
     (string-append
      "'s|'t|'re|'ve|'m|'ll|'d|"
      " ?\\p{L}+|"
      " ?\\p{N}+|"
      " ?(?:(?!\\p{L})(?!\\p{N})(?!" ws ").)+|"
      ws "+(?!" not-ws ")|"
      ws "+"))))

(define (gpt2-pretokenize text)
  (unless (string? text)
    (raise-argument-error 'gpt2-pretokenize "string?" text))
  (regexp-match* gpt2-pretoken-pattern text))
