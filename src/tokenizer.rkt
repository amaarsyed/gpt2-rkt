#lang racket/base

;; GPT-2 byte-level BPE. The byte map and merges follow encoder.py.
;; <|endoftext|> is matched on the raw text before the pre-tokenizer,
;; because the regex would otherwise split it. That special-token scan
;; is tiktoken and HF behaviour. A prompt that contains the literal
;; <|endoftext|> therefore encodes as token 50256.

(require json
         racket/port
         "bpe-bytes.rkt"
         "errors.rkt"
         "merges.rkt"
         "pretokenize.rkt")

(provide load-tokenizer
         make-tokenizer
         tokenizer?
         encode
         decode
         endoftext-token
         endoftext-id)

(define endoftext-token "<|endoftext|>")
(define endoftext-id 50256)

(struct tokenizer (token->id id->token ranks cache)
  #:constructor-name raw-tokenizer)

(define (make-tokenizer token->id ranks)
  (unless (hash? token->id)
    (raise-argument-error 'make-tokenizer "hash?" token->id))
  (unless (hash? ranks)
    (raise-argument-error 'make-tokenizer "hash?" ranks))
  (define id->token
    (for/hash ([(token id) (in-hash token->id)])
      (unless (and (string? token) (exact-nonnegative-integer? id))
        (raise-arguments-error 'make-tokenizer
                               "vocabulary entries must map strings to ids"
                               "token" token
                               "id" id))
      (values id token)))
  (raw-tokenizer token->id id->token ranks (make-hash)))

(define (read-utf8 path)
  (bytes->string/utf-8
   (call-with-input-file path port->bytes #:mode 'binary)))

(define (read-vocab path)
  (define parsed (read-json (open-input-string (read-utf8 path))))
  (unless (hash? parsed)
    (raise-gpt2 (format "vocabulary is not a JSON object: ~a" path)))
  (for/hash ([(key id) (in-hash parsed)])
    (define token (if (string? key) key (symbol->string key)))
    (unless (exact-nonnegative-integer? id)
      (raise-gpt2 (format "vocabulary id for ~a is not an integer" token)))
    (values token id)))

(define (load-tokenizer vocab-path merges-path)
  (unless (file-exists? vocab-path)
    (raise-gpt2 (format "missing vocabulary: ~a" vocab-path)))
  (unless (file-exists? merges-path)
    (raise-gpt2 (format "missing merges: ~a" merges-path)))
  (make-tokenizer (read-vocab vocab-path)
                  (read-merge-ranks (read-utf8 merges-path))))

(define (lowest-pair word ranks)
  (let loop ([word word] [best #f] [best-rank #f])
    (if (or (null? word) (null? (cdr word)))
        best
        (let* ([pair (cons (car word) (cadr word))]
               [rank (hash-ref ranks pair #f)])
          (if (and rank (or (not best-rank) (< rank best-rank)))
              (loop (cdr word) pair rank)
              (loop (cdr word) best best-rank))))))

;; Merge every non-overlapping occurrence of the pair, left to right.
(define (apply-merge word first second)
  (let loop ([word word] [out '()])
    (cond
      [(null? word) (reverse out)]
      [(and (equal? (car word) first)
            (pair? (cdr word))
            (equal? (cadr word) second))
       (loop (cddr word) (cons (string-append first second) out))]
      [else
       (loop (cdr word) (cons (car word) out))])))

(define (bpe-pieces token ranks cache)
  (define cached (hash-ref cache token #f))
  (cond
    [cached cached]
    [else
     (define word
       (for/list ([c (in-string token)])
         (string c)))
     (cond
       [(or (null? word) (null? (cdr word))) word]
       [else
        (define final
          (let loop ([word word])
            (define pair (lowest-pair word ranks))
            (cond
              [(not pair) word]
              [else
               (define merged (apply-merge word (car pair) (cdr pair)))
               (if (null? (cdr merged))
                   merged
                   (loop merged))])))
        (hash-set! cache token final)
        final])]))

(define (byte-encode text)
  (define bs (string->bytes/utf-8 text))
  (define n (bytes-length bs))
  (define out (make-string n))
  (for ([i (in-range n)])
    (string-set! out i (bpe-byte->char (bytes-ref bs i))))
  out)

(define (lookup-id tok piece)
  (hash-ref (tokenizer-token->id tok) piece
            (lambda ()
              (raise-arguments-error 'encode
                                     "token is not in the vocabulary"
                                     "token" piece))))

(define (encode-plain tok text)
  (define ranks (tokenizer-ranks tok))
  (define cache (tokenizer-cache tok))
  (let loop ([pieces (gpt2-pretokenize text)] [out '()])
    (cond
      [(null? pieces) (reverse out)]
      [else
       (define symbols (bpe-pieces (byte-encode (car pieces)) ranks cache))
       (loop (cdr pieces)
             (let add ([symbols symbols] [out out])
               (if (null? symbols)
                   out
                   (add (cdr symbols)
                        (cons (lookup-id tok (car symbols)) out)))))])))

(define (find-endoftext text start)
  (define needle endoftext-token)
  (define n (string-length needle))
  (define limit (string-length text))
  (let loop ([i start])
    (cond
      [(> (+ i n) limit) #f]
      [(for/and ([j (in-range n)])
         (char=? (string-ref text (+ i j)) (string-ref needle j)))
       i]
      [else (loop (add1 i))])))

(define (encode tok text)
  (unless (tokenizer? tok)
    (raise-argument-error 'encode "tokenizer?" tok))
  (unless (string? text)
    (raise-argument-error 'encode "string?" text))
  (let loop ([start 0])
    (define found (find-endoftext text start))
    (cond
      [(not found)
       (encode-plain tok (substring text start))]
      [else
       (define id (lookup-id tok endoftext-token))
       (append (encode-plain tok (substring text start found))
               (cons id (loop (+ found (string-length endoftext-token)))))])))

(define (decode tok ids)
  (unless (tokenizer? tok)
    (raise-argument-error 'decode "tokenizer?" tok))
  (unless (list? ids)
    (raise-argument-error 'decode "list?" ids))
  (define text
    (let ([out (open-output-string)])
      (for ([id (in-list ids)])
        (unless (exact-nonnegative-integer? id)
          (raise-argument-error 'decode "exact-nonnegative-integer?" id))
        (display (hash-ref (tokenizer-id->token tok) id
                           (lambda ()
                             (raise-gpt2 (format "unknown token id: ~a" id))))
                 out))
      (get-output-string out)))
  (define n (string-length text))
  (define bs (make-bytes n))
  (for ([i (in-range n)])
    (bytes-set! bs i (bpe-char->byte (string-ref text i))))
  (bytes->string/utf-8 bs #\uFFFD))
