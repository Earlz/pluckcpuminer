/*
 * Copyright 2009 Colin Percival, 2011 ArtForz, 2011-2014 pooler, 2015 Jordan Earls
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include "cpuminer-config.h"
#include "miner.h"

#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <emmintrin.h>


#define USE_SSE2 1

#define ROTL(a, b) (((a) << (b)) | ((a) >> (32 - (b))))
//note, this is 64 bytes
static inline void xor_salsa8(uint32_t B[16], const uint32_t Bx[16])
{
    uint32_t x00,x01,x02,x03,x04,x05,x06,x07,x08,x09,x10,x11,x12,x13,x14,x15;
    int i;

    x00 = (B[ 0] ^= Bx[ 0]);
    x01 = (B[ 1] ^= Bx[ 1]);
    x02 = (B[ 2] ^= Bx[ 2]);
    x03 = (B[ 3] ^= Bx[ 3]);
    x04 = (B[ 4] ^= Bx[ 4]);
    x05 = (B[ 5] ^= Bx[ 5]);
    x06 = (B[ 6] ^= Bx[ 6]);
    x07 = (B[ 7] ^= Bx[ 7]);
    x08 = (B[ 8] ^= Bx[ 8]);
    x09 = (B[ 9] ^= Bx[ 9]);
    x10 = (B[10] ^= Bx[10]);
    x11 = (B[11] ^= Bx[11]);
    x12 = (B[12] ^= Bx[12]);
    x13 = (B[13] ^= Bx[13]);
    x14 = (B[14] ^= Bx[14]);
    x15 = (B[15] ^= Bx[15]);
    for (i = 0; i < 8; i += 2) {
        /* Operate on columns. */
        x04 ^= ROTL(x00 + x12,  7);  x09 ^= ROTL(x05 + x01,  7);
        x14 ^= ROTL(x10 + x06,  7);  x03 ^= ROTL(x15 + x11,  7);

        x08 ^= ROTL(x04 + x00,  9);  x13 ^= ROTL(x09 + x05,  9);
        x02 ^= ROTL(x14 + x10,  9);  x07 ^= ROTL(x03 + x15,  9);

        x12 ^= ROTL(x08 + x04, 13);  x01 ^= ROTL(x13 + x09, 13);
        x06 ^= ROTL(x02 + x14, 13);  x11 ^= ROTL(x07 + x03, 13);

        x00 ^= ROTL(x12 + x08, 18);  x05 ^= ROTL(x01 + x13, 18);
        x10 ^= ROTL(x06 + x02, 18);  x15 ^= ROTL(x11 + x07, 18);

        /* Operate on rows. */
        x01 ^= ROTL(x00 + x03,  7);  x06 ^= ROTL(x05 + x04,  7);
        x11 ^= ROTL(x10 + x09,  7);  x12 ^= ROTL(x15 + x14,  7);

        x02 ^= ROTL(x01 + x00,  9);  x07 ^= ROTL(x06 + x05,  9);
        x08 ^= ROTL(x11 + x10,  9);  x13 ^= ROTL(x12 + x15,  9);

        x03 ^= ROTL(x02 + x01, 13);  x04 ^= ROTL(x07 + x06, 13);
        x09 ^= ROTL(x08 + x11, 13);  x14 ^= ROTL(x13 + x12, 13);

        x00 ^= ROTL(x03 + x02, 18);  x05 ^= ROTL(x04 + x07, 18);
        x10 ^= ROTL(x09 + x08, 18);  x15 ^= ROTL(x14 + x13, 18);
    }
    B[ 0] += x00;
    B[ 1] += x01;
    B[ 2] += x02;
    B[ 3] += x03;
    B[ 4] += x04;
    B[ 5] += x05;
    B[ 6] += x06;
    B[ 7] += x07;
    B[ 8] += x08;
    B[ 9] += x09;
    B[10] += x10;
    B[11] += x11;
    B[12] += x12;
    B[13] += x13;
    B[14] += x14;
    B[15] += x15;
}

#ifdef USE_SSE2

static inline void xor_salsa8_sse2(__m128i B[4], const __m128i Bx[4])
{
  __m128i X0, X1, X2, X3;
  __m128i T;
  int i;

  X0 = B[0] = _mm_xor_si128(B[0], Bx[0]);
  X1 = B[1] = _mm_xor_si128(B[1], Bx[1]);
  X2 = B[2] = _mm_xor_si128(B[2], Bx[2]);
  X3 = B[3] = _mm_xor_si128(B[3], Bx[3]);

  for (i = 0; i < 8; i += 2) {
    /* Operate on "columns". */
    T = _mm_add_epi32(X0, X3);
    X1 = _mm_xor_si128(X1, _mm_slli_epi32(T, 7));
    X1 = _mm_xor_si128(X1, _mm_srli_epi32(T, 25));
    T = _mm_add_epi32(X1, X0);
    X2 = _mm_xor_si128(X2, _mm_slli_epi32(T, 9));
    X2 = _mm_xor_si128(X2, _mm_srli_epi32(T, 23));
    T = _mm_add_epi32(X2, X1);
    X3 = _mm_xor_si128(X3, _mm_slli_epi32(T, 13));
    X3 = _mm_xor_si128(X3, _mm_srli_epi32(T, 19));
    T = _mm_add_epi32(X3, X2);
    X0 = _mm_xor_si128(X0, _mm_slli_epi32(T, 18));
    X0 = _mm_xor_si128(X0, _mm_srli_epi32(T, 14));

    /* Rearrange data. */
    X1 = _mm_shuffle_epi32(X1, 0x93);
    X2 = _mm_shuffle_epi32(X2, 0x4E);
    X3 = _mm_shuffle_epi32(X3, 0x39);

    /* Operate on "rows". */
    T = _mm_add_epi32(X0, X1);
    X3 = _mm_xor_si128(X3, _mm_slli_epi32(T, 7));
    X3 = _mm_xor_si128(X3, _mm_srli_epi32(T, 25));
    T = _mm_add_epi32(X3, X0);
    X2 = _mm_xor_si128(X2, _mm_slli_epi32(T, 9));
    X2 = _mm_xor_si128(X2, _mm_srli_epi32(T, 23));
    T = _mm_add_epi32(X2, X3);
    X1 = _mm_xor_si128(X1, _mm_slli_epi32(T, 13));
    X1 = _mm_xor_si128(X1, _mm_srli_epi32(T, 19));
    T = _mm_add_epi32(X1, X2);
    X0 = _mm_xor_si128(X0, _mm_slli_epi32(T, 18));
    X0 = _mm_xor_si128(X0, _mm_srli_epi32(T, 14));

    /* Rearrange data. */
    X1 = _mm_shuffle_epi32(X1, 0x39);
    X2 = _mm_shuffle_epi32(X2, 0x4E);
    X3 = _mm_shuffle_epi32(X3, 0x93);
  }

  B[0] = _mm_add_epi32(B[0], X0);
  B[1] = _mm_add_epi32(B[1], X1);
  B[2] = _mm_add_epi32(B[2], X2);
  B[3] = _mm_add_epi32(B[3], X3);
}

#endif


/*
static inline void xor_salsa8(uint32_t B[16], const uint32_t Bx[16])
{
  uint32_t x00,x01,x02,x03,x04,x05,x06,x07,x08,x09,x10,x11,x12,x13,x14,x15;
  int i;

  x00 = (B[ 0] ^= Bx[ 0]);
  x01 = (B[ 1] ^= Bx[ 1]);
  x02 = (B[ 2] ^= Bx[ 2]);
  x03 = (B[ 3] ^= Bx[ 3]);
  x04 = (B[ 4] ^= Bx[ 4]);
  x05 = (B[ 5] ^= Bx[ 5]);
  x06 = (B[ 6] ^= Bx[ 6]);
  x07 = (B[ 7] ^= Bx[ 7]);
  x08 = (B[ 8] ^= Bx[ 8]);
  x09 = (B[ 9] ^= Bx[ 9]);
  x10 = (B[10] ^= Bx[10]);
  x11 = (B[11] ^= Bx[11]);
  x12 = (B[12] ^= Bx[12]);
  x13 = (B[13] ^= Bx[13]);
  x14 = (B[14] ^= Bx[14]);
  x15 = (B[15] ^= Bx[15]);
  for (i = 0; i < 8; i += 2) {
#define R(a, b) (((a) << (b)) | ((a) >> (32 - (b))))
    /* Operate on columns. *
    x04 ^= R(x00+x12, 7); x09 ^= R(x05+x01, 7);
    x14 ^= R(x10+x06, 7); x03 ^= R(x15+x11, 7);
    
    x08 ^= R(x04+x00, 9); x13 ^= R(x09+x05, 9);
    x02 ^= R(x14+x10, 9); x07 ^= R(x03+x15, 9);
    
    x12 ^= R(x08+x04,13); x01 ^= R(x13+x09,13);
    x06 ^= R(x02+x14,13); x11 ^= R(x07+x03,13);
    
    x00 ^= R(x12+x08,18); x05 ^= R(x01+x13,18);
    x10 ^= R(x06+x02,18); x15 ^= R(x11+x07,18);
    
    /* Operate on rows. *
    x01 ^= R(x00+x03, 7); x06 ^= R(x05+x04, 7);
    x11 ^= R(x10+x09, 7); x12 ^= R(x15+x14, 7);
    
    x02 ^= R(x01+x00, 9); x07 ^= R(x06+x05, 9);
    x08 ^= R(x11+x10, 9); x13 ^= R(x12+x15, 9);
    
    x03 ^= R(x02+x01,13); x04 ^= R(x07+x06,13);
    x09 ^= R(x08+x11,13); x14 ^= R(x13+x12,13);
    
    x00 ^= R(x03+x02,18); x05 ^= R(x04+x07,18);
    x10 ^= R(x09+x08,18); x15 ^= R(x14+x13,18);
#undef R
  }
  B[ 0] += x00;
  B[ 1] += x01;
  B[ 2] += x02;
  B[ 3] += x03;
  B[ 4] += x04;
  B[ 5] += x05;
  B[ 6] += x06;
  B[ 7] += x07;
  B[ 8] += x08;
  B[ 9] += x09;
  B[10] += x10;
  B[11] += x11;
  B[12] += x12;
  B[13] += x13;
  B[14] += x14;
  B[15] += x15;
}
*/

uint32_t static inline ReadBE32(const unsigned char* ptr)
{
    return be32toh(*((uint32_t*)ptr));
}

uint64_t static inline ReadBE64(const unsigned char* ptr)
{
    return be64toh(*((uint64_t*)ptr));
}

void static inline WriteBE32(unsigned char* ptr, uint32_t x)
{
    *((uint32_t*)ptr) = htobe32(x);
}
void static inline WriteBE64(unsigned char* ptr, uint64_t x)
{
    *((uint64_t*)ptr) = htobe64(x);
}
//reimplement sha256 for now with this easy to use, but unoptimized version

uint32_t static inline Ch(uint32_t x, uint32_t y, uint32_t z) { return z ^ (x & (y ^ z)); }
uint32_t static inline Maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (z & (x | y)); }
uint32_t static inline Sigma0(uint32_t x) { return (x >> 2 | x << 30) ^ (x >> 13 | x << 19) ^ (x >> 22 | x << 10); }
uint32_t static inline Sigma1(uint32_t x) { return (x >> 6 | x << 26) ^ (x >> 11 | x << 21) ^ (x >> 25 | x << 7); }
uint32_t static inline sigma0(uint32_t x) { return (x >> 7 | x << 25) ^ (x >> 18 | x << 14) ^ (x >> 3); }
uint32_t static inline sigma1(uint32_t x) { return (x >> 17 | x << 15) ^ (x >> 19 | x << 13) ^ (x >> 10); }

/** One round of SHA-256. */
void static inline Round(uint32_t a, uint32_t b, uint32_t c, uint32_t* d, uint32_t e, uint32_t f, uint32_t g, uint32_t* h, uint32_t k, uint32_t w)
{
    uint32_t t1 = *h + Sigma1(e) + Ch(e, f, g) + k + w;
    uint32_t t2 = Sigma0(a) + Maj(a, b, c);
    *d += t1;
    *h = t1 + t2;
}

/** Initialize SHA-256 state. */
void static inline Initialize(uint32_t* s)
{
    s[0] = 0x6a09e667ul;
    s[1] = 0xbb67ae85ul;
    s[2] = 0x3c6ef372ul;
    s[3] = 0xa54ff53aul;
    s[4] = 0x510e527ful;
    s[5] = 0x9b05688cul;
    s[6] = 0x1f83d9abul;
    s[7] = 0x5be0cd19ul;
}

/** Perform one SHA-256 transformation, processing a 64-byte chunk. */
void Transform(uint32_t* s, const unsigned char* chunk)
{
    uint32_t a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5], g = s[6], h = s[7];
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15;

    Round(a, b, c, &d, e, f, g, &h, 0x428a2f98, w0 = ReadBE32(chunk + 0));
    Round(h, a, b, &c, d, e, f, &g, 0x71374491, w1 = ReadBE32(chunk + 4));
    Round(g, h, a, &b, c, d, e, &f, 0xb5c0fbcf, w2 = ReadBE32(chunk + 8));
    Round(f, g, h, &a, b, c, d, &e, 0xe9b5dba5, w3 = ReadBE32(chunk + 12));
    Round(e, f, g, &h, a, b, c, &d, 0x3956c25b, w4 = ReadBE32(chunk + 16));
    Round(d, e, f, &g, h, a, b, &c, 0x59f111f1, w5 = ReadBE32(chunk + 20));
    Round(c, d, e, &f, g, h, a, &b, 0x923f82a4, w6 = ReadBE32(chunk + 24));
    Round(b, c, d, &e, f, g, h, &a, 0xab1c5ed5, w7 = ReadBE32(chunk + 28));
    Round(a, b, c, &d, e, f, g, &h, 0xd807aa98, w8 = ReadBE32(chunk + 32));
    Round(h, a, b, &c, d, e, f, &g, 0x12835b01, w9 = ReadBE32(chunk + 36));
    Round(g, h, a, &b, c, d, e, &f, 0x243185be, w10 = ReadBE32(chunk + 40));
    Round(f, g, h, &a, b, c, d, &e, 0x550c7dc3, w11 = ReadBE32(chunk + 44));
    Round(e, f, g, &h, a, b, c, &d, 0x72be5d74, w12 = ReadBE32(chunk + 48));
    Round(d, e, f, &g, h, a, b, &c, 0x80deb1fe, w13 = ReadBE32(chunk + 52));
    Round(c, d, e, &f, g, h, a, &b, 0x9bdc06a7, w14 = ReadBE32(chunk + 56));
    Round(b, c, d, &e, f, g, h, &a, 0xc19bf174, w15 = ReadBE32(chunk + 60));

    Round(a, b, c, &d, e, f, g, &h, 0xe49b69c1, w0 += sigma1(w14) + w9 + sigma0(w1));
    Round(h, a, b, &c, d, e, f, &g, 0xefbe4786, w1 += sigma1(w15) + w10 + sigma0(w2));
    Round(g, h, a, &b, c, d, e, &f, 0x0fc19dc6, w2 += sigma1(w0) + w11 + sigma0(w3));
    Round(f, g, h, &a, b, c, d, &e, 0x240ca1cc, w3 += sigma1(w1) + w12 + sigma0(w4));
    Round(e, f, g, &h, a, b, c, &d, 0x2de92c6f, w4 += sigma1(w2) + w13 + sigma0(w5));
    Round(d, e, f, &g, h, a, b, &c, 0x4a7484aa, w5 += sigma1(w3) + w14 + sigma0(w6));
    Round(c, d, e, &f, g, h, a, &b, 0x5cb0a9dc, w6 += sigma1(w4) + w15 + sigma0(w7));
    Round(b, c, d, &e, f, g, h, &a, 0x76f988da, w7 += sigma1(w5) + w0 + sigma0(w8));
    Round(a, b, c, &d, e, f, g, &h, 0x983e5152, w8 += sigma1(w6) + w1 + sigma0(w9));
    Round(h, a, b, &c, d, e, f, &g, 0xa831c66d, w9 += sigma1(w7) + w2 + sigma0(w10));
    Round(g, h, a, &b, c, d, e, &f, 0xb00327c8, w10 += sigma1(w8) + w3 + sigma0(w11));
    Round(f, g, h, &a, b, c, d, &e, 0xbf597fc7, w11 += sigma1(w9) + w4 + sigma0(w12));
    Round(e, f, g, &h, a, b, c, &d, 0xc6e00bf3, w12 += sigma1(w10) + w5 + sigma0(w13));
    Round(d, e, f, &g, h, a, b, &c, 0xd5a79147, w13 += sigma1(w11) + w6 + sigma0(w14));
    Round(c, d, e, &f, g, h, a, &b, 0x06ca6351, w14 += sigma1(w12) + w7 + sigma0(w15));
    Round(b, c, d, &e, f, g, h, &a, 0x14292967, w15 += sigma1(w13) + w8 + sigma0(w0));

    Round(a, b, c, &d, e, f, g, &h, 0x27b70a85, w0 += sigma1(w14) + w9 + sigma0(w1));
    Round(h, a, b, &c, d, e, f, &g, 0x2e1b2138, w1 += sigma1(w15) + w10 + sigma0(w2));
    Round(g, h, a, &b, c, d, e, &f, 0x4d2c6dfc, w2 += sigma1(w0) + w11 + sigma0(w3));
    Round(f, g, h, &a, b, c, d, &e, 0x53380d13, w3 += sigma1(w1) + w12 + sigma0(w4));
    Round(e, f, g, &h, a, b, c, &d, 0x650a7354, w4 += sigma1(w2) + w13 + sigma0(w5));
    Round(d, e, f, &g, h, a, b, &c, 0x766a0abb, w5 += sigma1(w3) + w14 + sigma0(w6));
    Round(c, d, e, &f, g, h, a, &b, 0x81c2c92e, w6 += sigma1(w4) + w15 + sigma0(w7));
    Round(b, c, d, &e, f, g, h, &a, 0x92722c85, w7 += sigma1(w5) + w0 + sigma0(w8));
    Round(a, b, c, &d, e, f, g, &h, 0xa2bfe8a1, w8 += sigma1(w6) + w1 + sigma0(w9));
    Round(h, a, b, &c, d, e, f, &g, 0xa81a664b, w9 += sigma1(w7) + w2 + sigma0(w10));
    Round(g, h, a, &b, c, d, e, &f, 0xc24b8b70, w10 += sigma1(w8) + w3 + sigma0(w11));
    Round(f, g, h, &a, b, c, d, &e, 0xc76c51a3, w11 += sigma1(w9) + w4 + sigma0(w12));
    Round(e, f, g, &h, a, b, c, &d, 0xd192e819, w12 += sigma1(w10) + w5 + sigma0(w13));
    Round(d, e, f, &g, h, a, b, &c, 0xd6990624, w13 += sigma1(w11) + w6 + sigma0(w14));
    Round(c, d, e, &f, g, h, a, &b, 0xf40e3585, w14 += sigma1(w12) + w7 + sigma0(w15));
    Round(b, c, d, &e, f, g, h, &a, 0x106aa070, w15 += sigma1(w13) + w8 + sigma0(w0));

    Round(a, b, c, &d, e, f, g, &h, 0x19a4c116, w0 += sigma1(w14) + w9 + sigma0(w1));
    Round(h, a, b, &c, d, e, f, &g, 0x1e376c08, w1 += sigma1(w15) + w10 + sigma0(w2));
    Round(g, h, a, &b, c, d, e, &f, 0x2748774c, w2 += sigma1(w0) + w11 + sigma0(w3));
    Round(f, g, h, &a, b, c, d, &e, 0x34b0bcb5, w3 += sigma1(w1) + w12 + sigma0(w4));
    Round(e, f, g, &h, a, b, c, &d, 0x391c0cb3, w4 += sigma1(w2) + w13 + sigma0(w5));
    Round(d, e, f, &g, h, a, b, &c, 0x4ed8aa4a, w5 += sigma1(w3) + w14 + sigma0(w6));
    Round(c, d, e, &f, g, h, a, &b, 0x5b9cca4f, w6 += sigma1(w4) + w15 + sigma0(w7));
    Round(b, c, d, &e, f, g, h, &a, 0x682e6ff3, w7 += sigma1(w5) + w0 + sigma0(w8));
    Round(a, b, c, &d, e, f, g, &h, 0x748f82ee, w8 += sigma1(w6) + w1 + sigma0(w9));
    Round(h, a, b, &c, d, e, f, &g, 0x78a5636f, w9 += sigma1(w7) + w2 + sigma0(w10));
    Round(g, h, a, &b, c, d, e, &f, 0x84c87814, w10 += sigma1(w8) + w3 + sigma0(w11));
    Round(f, g, h, &a, b, c, d, &e, 0x8cc70208, w11 += sigma1(w9) + w4 + sigma0(w12));
    Round(e, f, g, &h, a, b, c, &d, 0x90befffa, w12 += sigma1(w10) + w5 + sigma0(w13));
    Round(d, e, f, &g, h, a, b, &c, 0xa4506ceb, w13 += sigma1(w11) + w6 + sigma0(w14));
    Round(c, d, e, &f, g, h, a, &b, 0xbef9a3f7, w14 + sigma1(w12) + w7 + sigma0(w15));
    Round(b, c, d, &e, f, g, h, &a, 0xc67178f2, w15 + sigma1(w13) + w8 + sigma0(w0));

    s[0] += a;
    s[1] += b;
    s[2] += c;
    s[3] += d;
    s[4] += e;
    s[5] += f;
    s[6] += g;
    s[7] += h;
}

typedef struct sha256state_s
{
    uint32_t s[8];
    unsigned char buf[64];
    size_t bytes;
} sha256state;

#define OUTPUT_SIZE 32
////// SHA-256

void sha256init(sha256state* state)
{
    //memset(state, 0, sizeof(sha256state));
    state->bytes=0;
    Initialize(state->s);
}

void sha256Write(sha256state *state, const unsigned char* data, size_t len)
{
    const unsigned char* end = data + len;
    size_t bufsize = state->bytes % 64;
    if (bufsize && bufsize + len >= 64) {
        // Fill the buffer, and process it.
        memcpy(state->buf + bufsize, data, 64 - bufsize);
        state->bytes += 64 - bufsize;
        data += 64 - bufsize;
        Transform(state->s, state->buf);
        bufsize = 0;
    }
    while (end >= data + 64) {
        // Process full chunks directly from the source.
        Transform(state->s, data);
        state->bytes += 64;
        data += 64;
    }
    if (end > data) {
        // Fill the buffer with what remains.
        memcpy(state->buf + bufsize, data, end - data);
        state->bytes += end - data;
    }
}

void sha256Finalize(sha256state *state,unsigned char hash[OUTPUT_SIZE])
{
    static const unsigned char pad[64] = {0x80};
    unsigned char sizedesc[8];
    WriteBE64(sizedesc, state->bytes << 3);
    sha256Write(state, pad, 1 + ((119 - (state->bytes % 64)) % 64));
    sha256Write(state, sizedesc, 8);
    WriteBE32(hash, state->s[0]);
    WriteBE32(hash + 4, state->s[1]);
    WriteBE32(hash + 8, state->s[2]);
    WriteBE32(hash + 12, state->s[3]);
    WriteBE32(hash + 16, state->s[4]);
    WriteBE32(hash + 20, state->s[5]);
    WriteBE32(hash + 24, state->s[6]);
    WriteBE32(hash + 28, state->s[7]);
}

void sha256Reset(sha256state* state)
{
    state->bytes = 0;
    sha256init(state);
}

static inline void assert(int cond)
{
  if(!cond)
  {
    printf("error\n");
    exit(1);
  }
}
//computes a single sha256 hash
void sha256_hash(unsigned char *hash, const unsigned char *data, int len)
{
  uint32_t S[16] __attribute__((aligned(32))), T[16] __attribute__((aligned(32)));
  int i, r;

  sha256_init(S);
  for (r = len; r > -9; r -= 64) {
    if (r < 64)
      memset(T, 0, 64);
    memcpy(T, data + len - r, r > 64 ? 64 : (r < 0 ? 0 : r));
    if (r >= 0 && r < 64)
      ((unsigned char *)T)[r] = 0x80;
    for (i = 0; i < 16; i++)
      T[i] = be32dec(T + i);
    if (r < 56)
      T[15] = 8 * len;
    sha256_transform(S, T, 0);
  }
  for (i = 0; i < 8; i++)
    be32enc((uint32_t *)hash + i, S[i]);
}

//hash exactly 64 bytes (ie, sha256 block size)
void sha256_hash512(unsigned char *hash, const unsigned char *data)
{
  uint32_t S[16] __attribute__((aligned(32))), T[16] __attribute__((aligned(32)));
  int i, r;

  sha256_init(S);

    memcpy(T, data, 64);
    for (i = 0; i < 16; i++)
      T[i] = be32dec(T + i);
    sha256_transform(S, T, 0);

    memset(T, 0, 64);
    //memcpy(T, data + 64, 0);
    ((unsigned char *)T)[0] = 0x80;
    for (i = 0; i < 16; i++)
      T[i] = be32dec(T + i);
      T[15] = 8 * 64;
    sha256_transform(S, T, 0);

  for (i = 0; i < 8; i++)
    be32enc((uint32_t *)hash + i, S[i]);
}

//static const int HASH_MEMORY=128*1024;

int scanhash_pluck(int thr_id, uint32_t *pdata,
  unsigned char *scratchbuf, const uint32_t *ptarget,
  uint32_t max_nonce, unsigned long *hashes_done, int N)
{
  uint32_t data[20], hash[8];
  uint32_t S[16];
  //uint32_t midstate[8];
  uint32_t n = pdata[19] - 1;
  const int HASH_MEMORY=N*1024;
  const uint32_t first_nonce = pdata[19];
  const uint32_t Htarg = ptarget[7];
  int throughput = 1;
  int counti;
  
  
  //for (counti = 0; counti < throughput; counti++)
    memcpy(data, pdata, 80);

     for (int a = 0; a < 20; a++)
      be32enc((uint32_t *)data + a, (((uint32_t*)pdata))[a]); 

 // memcpy(S + 8, pdata + 8, 32);
  //sha256_init(midstate);
  //sha256_transform(midstate, data, 0);
    printf("data: ");
      for(int meh=0;meh<80;meh++)
    {
        printf("%x", ((uint8_t*)data)[meh]);
    }
    printf("\n");
  do {
    //for (counti = 0; counti < throughput; counti++)
      data[19] = ++n; //incrementing nonce (?)

const int BLOCK_HEADER_SIZE=80;
    //could probably cache this so that we can skip hash generation when the first sha256 hash matches
    uint8_t *hashbuffer = scratchbuf; //don't allocate this on stack, since it's huge.. 
    //allocating on heap adds hardly any overhead on Linux
    int size=HASH_MEMORY;
    sha256state sha;
    //uint8_t buffer[HASH_MEMORY];
    memset(hashbuffer, 0, 64); 
    //memcpy(hashbuffer, data.begin(), 32); 
    sha256_hash(&hashbuffer[0], data, BLOCK_HEADER_SIZE);
    /*
    sha256Reset(&sha);
    sha256Write(&sha, data, BLOCK_HEADER_SIZE);
    sha256Finalize(&sha, &hashbuffer[0]);
    */
    

    for (int i = 64; i < size-32; i+=32)
    {
        //i-4 because we use integers for all references against this, and we don't want to go 3 bytes over the defined area
        int randmax = i-4; //we could use size here, but then it's probable to use 0 as the value in most cases
        uint32_t joint[16];
        uint32_t randbuffer[16];
        assert(i-32>0);
        assert(i<size);
        uint32_t randseed[16];
        assert(sizeof(int)*16 == 64);

        //setup randbuffer to be an array of random indexes
        memcpy(randseed, &hashbuffer[i-64], 64);
        if(i>128)
        {
            memcpy(randbuffer, &hashbuffer[i-128], 64);
        }else
        {
            memset(&randbuffer, 0, 64);
        }
        xor_salsa8(randbuffer, randseed);

        memcpy(joint, &hashbuffer[i-32], 32);
        //use the last hash value as the seed
        for (int j = 32; j < 64; j+=4)
        {
            assert((j - 32) / 2 < 16);
            //every other time, change to next random index
            uint32_t rand = randbuffer[(j - 32)/4] % (randmax-32); //randmax - 32 as otherwise we go beyond memory that's already been written to
            assert(j>0 && j<64);
            assert(rand<size);
            assert(j/2 < 64);
            joint[j/4] = *((uint32_t*)&hashbuffer[rand]);
        }
        assert(i>=0 && i+32<size);

        sha256_hash512(&hashbuffer[i], joint);

        //setup randbuffer to be an array of random indexes
        memcpy(randseed, &hashbuffer[i-32], 64); //use last hash value and previous hash value(post-mixing)
        if(i>128)
        {
            memcpy(randbuffer, &hashbuffer[i-128], 64);
        }else
        {
            memset(randbuffer, 0, 64);
        }
        xor_salsa8(randbuffer, randseed);

        //use the last hash value as the seed
        for (int j = 0; j < 32; j+=2)
        {
            assert(j/4 < 16);
            uint32_t rand = randbuffer[j/2] % randmax;
            assert(rand < size);
            assert((j/4)+i >= 0 && (j/4)+i < size);
            assert(j + i -4 < i + 32); //assure we dont' access beyond the hash
            *((uint32_t*)&hashbuffer[rand]) = *((uint32_t*)&hashbuffer[j + i - 4]);
        }
    }
    //note: off-by-one error is likely here...     
    for (int i = size-64-1; i >= 64; i -= 64)       
    {      
      assert(i-64 >= 0);       
      assert(i+64<size);    
      sha256_hash512(&hashbuffer[i-64], &hashbuffer[i]);
    /*  sha256Reset(&sha);
      sha256Write(&sha, &hashbuffer[i], 64);
      sha256Finalize(&sha, &hashbuffer[i-64]); */
     }


    //for (int a = 0; a < 8; a++)
    //  hash[a] = swab32(((uint32_t*)hashbuffer)[a]);
    //for (int a = 0; a < 8; a++)
    //  be32enc((uint32_t *)hash + a, (((uint32_t*)hashbuffer))[a]);
    memcpy((unsigned char*)hash, hashbuffer, 32);
    //free(hashbuffer);
    //printf("hash: %u\n", (unsigned int)hash[0]);
      if (hash[7] <= Htarg && fulltest(hash, ptarget)) {
        *hashes_done = n - pdata[19] + 1;
        pdata[19] = htobe32(data[19]);
        printf("hasheddata:");
      for(int meh=0;meh<80;meh++)
    {
        printf("%x", ((uint8_t*)data)[meh]);
    }
            printf("\ntxxxeddata:");
      for(int meh=0;meh<80;meh++)
    {
        printf("%x", ((uint8_t*)pdata)[meh]);
    }
    printf("\nhash:");
      for(int meh=0;meh<32;meh++)
    {
        printf("%x", ((uint8_t*)hash)[meh]);
    }
    printf("\n");
        return 1;
      }
    //scrypt_1024_1_1_256(data, hash, midstate, scratchbuf, N);
     /* if (fulltest(hash, ptarget)) {
        *hashes_done = n - first_nonce + 1;
        return 1;
      }*/
  } while (n < max_nonce && !work_restart[thr_id].restart);
  
  *hashes_done = n - first_nonce + 1;
  pdata[19] = n;
  return 0;
}
