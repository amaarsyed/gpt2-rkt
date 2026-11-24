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
         "pretokenize.rkt"
         "source-map.rkt"
         "trace.rkt"
         "utf8-stream.rkt")

(provide load-tokenizer
         make-tokenizer
         tokenizer?
         encode
         decode
         make-token-decoder
         token-decoder-add
         token-decoder-finish
         token-decoder-text
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

;; info is #f on the fast path. A box receives
;; (hasheq 'cached? boolean 'merges (or list? 'null) 'pieces list?).
(define (bpe-pieces token ranks cache [info #f])
  (define cached (hash-ref cache token #f))
  (cond
    [cached
     (when info
       (set-box! info (hasheq 'cached? #t 'merges 'null 'pieces cached)))
     cached]
    [else
     (define word
       (for/list ([c (in-string token)])
         (string c)))
     (cond
       [(or (null? word) (null? (cdr word)))
        (when info
          (set-box! info (hasheq 'cached? #f 'merges '() 'pieces word)))
        word]
       [(eq? info #f)
        (define final
          (stage bpe-merge bpe-pieces
            (let loop ([word word])
              (define pair (lowest-pair word ranks))
              (cond
                [(not pair) word]
                [else
                 (define merged (apply-merge word (car pair) (cdr pair)))
                 (if (null? (cdr merged))
                     merged
                     (loop merged))]))))
        (hash-set! cache token final)
        final]
       [else
        (define-values (final steps)
          (let loop ([word word] [steps '()])
            (define pair (lowest-pair word ranks))
            (cond
              [(not pair) (values word (reverse steps))]
              [else
               (define rank (hash-ref ranks pair))
               (define merged (apply-merge word (car pair) (cdr pair)))
               (define step
                 (hasheq 'pair (list (car pair) (cdr pair))
                         'rank rank
                         'merged (string-append (car pair) (cdr pair))
                         'pieces merged))
               (if (null? (cdr merged))
                   (values merged (reverse (cons step steps)))
                   (loop merged (cons step steps)))])))
        (hash-set! cache token final)
        (set-box! info (hasheq 'cached? #f 'merges steps 'pieces final))
        final])]))

(define (encode-plain-fast tok text)
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

(define (record-bpe-chunk! obs tok raw)
  (define ranks (tokenizer-ranks tok))
  (define cache (tokenizer-cache tok))
  (define encoded (byte-encode raw))
  (define info (box #f))
  (define symbols (bpe-pieces encoded ranks cache info))
  (define got (unbox info))
  (define ids
    (for/list ([sym (in-list symbols)])
      (lookup-id tok sym)))
  (define strings
    (for/list ([id (in-list ids)])
      (decode tok (list id))))
  (trace-bpe-chunk!
   obs
   (hasheq 'text raw
           'byteEncoded encoded
           'cached (hash-ref got 'cached?)
           'merges (hash-ref got 'merges)
           'pieces (hash-ref got 'pieces)
           'tokenIds ids
           'tokenStrings strings))
  ids)

(define (encode-plain-traced tok text obs)
  (define t0 (current-inexact-monotonic-milliseconds))
  (define ids
    (let loop ([pieces (gpt2-pretokenize text)] [out '()])
      (if (null? pieces)
          (reverse out)
          (loop (cdr pieces)
                (append (reverse (record-bpe-chunk! obs tok (car pieces))) out)))))
  (trace-span! obs 'bpe-merge #f 'bpe
               (- (current-inexact-monotonic-milliseconds) t0))
  ids)

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
  (define obs (current-trace-observer))
  (if (eq? obs #f)
      (encode-plain-fast tok text)
      (encode-plain-traced tok text obs)))

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

(define (token-bytes tok id)
  (unless (exact-nonnegative-integer? id)
    (raise-argument-error 'decode "exact-nonnegative-integer?" id))
  (define text
    (hash-ref (tokenizer-id->token tok) id
              (lambda ()
                (raise-gpt2 (format "unknown token id: ~a" id)))))
  (define n (string-length text))
  (define bs (make-bytes n))
  (for ([i (in-range n)])
    (bytes-set! bs i (bpe-char->byte (string-ref text i))))
  bs)

(struct token-decoder (tok stream)
  #:constructor-name raw-token-decoder)

(define (make-token-decoder tok)
  (unless (tokenizer? tok)
    (raise-argument-error 'make-token-decoder "tokenizer?" tok))
  (raw-token-decoder tok (make-utf8-stream)))

(define (token-decoder-add dec id)
  (utf8-stream-add (token-decoder-stream dec)
                   (token-bytes (token-decoder-tok dec) id)))

(define (token-decoder-finish dec)
  (utf8-stream-finish (token-decoder-stream dec)))

(define (token-decoder-text dec)
  (utf8-stream-text (token-decoder-stream dec)))

(define (decode tok ids)
  (unless (tokenizer? tok)
    (raise-argument-error 'decode "tokenizer?" tok))
  (unless (list? ids)
    (raise-argument-error 'decode "list?" ids))
  (define dec (make-token-decoder tok))
  (for ([id (in-list ids)])
    (token-decoder-add dec id))
  (token-decoder-finish dec)
  (token-decoder-text dec))
