/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

static int unparsable_function_1(
    struct b* c,
    int32_t d,
    int32_t e,
    bool f,
    const uint8_t* g,
    int32_t h,
    int32_t i,
    int32_t j,
    uint8_t* k,
    int32_t l,
    uint8_t* m,
    uint8_t* n) {
  o* p = c->q;
  uint8_t* r = p->s;
  uint8_t* t = c->u;
  int32_t v = (p->w & (uint8_t)0xe0) >> 5u;
  int x = (p->w & (uint8_t)0x10) >> 4u;
  int32_t y;
  int32_t z;
  int aa;
  int32_t ab;
  int32_t ac;
  int32_t ad = 0;
  int32_t ae = 0;
  uint8_t* af;
  int32_t ag = p->ah;
  const char* ai;
  int aj;

  aj = ak(g, &y, &z, NULL);
  if (aj < 0) {
    return aj;
  }

  if (p->al != NULL && p->al[j]) {
    return d;
  }

  if (p->am == NULL) {
    p->am = g;
  }

  bool an = ((p->ao == ap) && (p->aq & 0x08u) && !p->ar);
  if (an) {
    if (p->as == NULL) {
      at("au.");
      return av;
    }
    if (p->as->aw == NULL) {
      at("ax.");
      return av;
    }
    ay* az = (ay*)p->as->aw;
    char* ba = az->ba;
    int32_t bb = sizeof(int32_t) + sizeof(int64_t) + p->bc * sizeof(int32_t);
    size_t bd = ap + p->bc * sizeof(int32_t);
    int32_t be;
    int64_t bf;
    be = *(int32_t*)(g + bd);
    bf = *(int64_t*)(g + bd + sizeof(int32_t));
    int32_t* bg = (int32_t*)(g + bd + sizeof(int32_t) + sizeof(int64_t));
    int32_t bh = bg[j];
    FILE* bi = NULL;
    if (az->bj) {
      char* bk = malloc(strlen(az->ba) + 1 + 8 + strlen(".bl") + 1);
      bl(bk, bm);
      sprintf(bk, "%s/%08X.bl", az->ba, be);
      bi = fopen(bk, "rb");
      free(bk);
      fseek(bi, i, SEEK_SET);
    } else {
      bi = fopen(ba, "rb");
      fseek(bi, bf + i, SEEK_SET);
    }
    size_t bn = fread(t, 1, bh, bi);
    fclose(bi);
    if ((int32_t)bn != bh) {
      at("bo.");
      return bp;
    }
    g = t;
    i = 0;
    h = bh;
  }

  if (f) {
    int bq = e ? y % p->br : d;
    if (!p->ar) {
      if (y + p->ao != z) {
        return bs;
      }
      if (z < p->ao + (j * p->br) + bq) {
        return bp;
      }
    }
    if (!an) {
      g += p->ao + j * p->br;
    }
    af = k + l;
    if (p->bt != NULL) {
      af = m;
    }
    aj = 0;
    switch (p->ar) {
      case bu:
        aj = bv(p->ah, p->am, af, bq);
        if (aj < 0) {
          at("bw");
          return bx;
        }
        break;
      case by:
        aj = bz(p->ah, af, bq);
        if (aj < 0) {
          at("bz");
          return bx;
        }
        break;
      case ca:
        memset(af, 0, bq);
        break;
      case cb:
        break;
      default:
        memcpy(af, g, bq);
    }
    if (p->bt != NULL) {
      cc cd;
      memcpy(&cd, p->ce, sizeof(cd));
      cd.in = m;
      cd.out = k + l;
      cd.size = d;
      cd.typesize = ag;
      cd.offset = l;
      cd.tid = c->cf;
      cd.ttmp = c->cg;
      cd.ttmp_nbytes = c->ch;
      cd.ctx = p;

      if (p->bt(&cd) != 0) {
        at("ci");
        return cj;
      }
    }
    return bq;
  }

  if (!an && (i <= 0 || i >= h)) {
    return bx;
  }

  g += i;
  h -= i;

  int ck = cl(r, 'd');

  if (((ck >= 0) && (cm(r, cn, 'd') != co)) || p->bt != NULL) {
    af = m;
  } else {
    af = k + l;
  }

  if (!x && !e && !p->cp) {
    aa = (int32_t)ag;
  } else {
    aa = 1;
  }

  ab = d / aa;
  if (ab == 0) {
    return -1;
  }
  for (int cq = 0; cq < aa; cq++) {
    if (h < (signed)sizeof(int32_t)) {
      return bp;
    }
    h -= sizeof(int32_t);
    cbytes = cr(g);
    if (cbytes > 0) {
      if (h < cbytes) {
        return bp;
      }
      h -= cbytes;
    }
    g += sizeof(int32_t);
    ad += (signed)sizeof(int32_t);

    if (cbytes == 0) {
      memset(af, 0, (unsigned int)ab);
      ac = ab;
    } else if (cbytes < 0) {
      uint8_t cs;

      if (h < (signed)sizeof(uint8_t)) {
        return bp;
      }
      h -= sizeof(uint8_t);

      cs = g[0];
      g += 1;
      ad += 1;

      if (cs & 0x1) {
        if (cbytes < -255) {
          return ct;
        }
        uint8_t cu = -cbytes;
        memset(af, cu, (unsigned int)ab);
      } else {
        at("cv - %d", cs);
        return ct;
      }
      ac = ab;
      cbytes = 0;
    } else if (cbytes == ab) {
      memcpy(af, g, (unsigned int)ab);
      ac = (int32_t)ab;
    } else {
      if (v == cw) {
        ac = cx(g, cbytes, af, (int)ab);
      }
#if defined(cy)
      else if (v == cz) {
        ac = da((char*)g, (size_t)cbytes, (char*)af, (size_t)ab);
      }
#endif
#if defined(db)
      else if (v == dc) {
        ac = dd((char*)g, (size_t)cbytes, (char*)af, (size_t)ab);
      }
#endif
#if defined(de)
      else if (v == df) {
        ac = dg((char*)g, (size_t)cbytes, (char*)af, (size_t)ab);
      }
#endif
#if defined(dh)
      else if (v == di) {
        ac = dj(c, (char*)g, (size_t)cbytes, (char*)af, (size_t)ab);
      }
#endif
      else {
        ai = dk(v);
        at("dl '%s' dm.  "
           "dn.",
           ai);
        return dp;
      }

      if (ac != ab) {
        return bx;
      }
    }
    g += cbytes;
    ad += cbytes;
    af += ac;
    ae += ac;
  }

  if (ck >= 0 || p->bt != NULL) {
    int dq = dr(c, d, k, l, m, n, t, ck);
    if (dq < 0)
      return dq;
  }

  return (int)ae;
}
