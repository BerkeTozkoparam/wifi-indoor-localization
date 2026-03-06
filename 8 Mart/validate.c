/*
 * validate.c — Kadın Güvenlik Haritası JSON Doğrulayıcı
 * 8 Mart Dünya Kadınlar Günü Projesi
 *
 * Kullanım: ./validate markers.json
 *
 * Derleme: gcc -o validate validate.c
 *
 * Kontrol edilen kurallar:
 *   1. Dosya açılabilir olmalı
 *   2. JSON dizisi "[" ile başlamalı, "]" ile bitmeli
 *   3. Her kayıt şu alanları içermeli:
 *      "lat", "lon", "durum", "kategori", "tarih"
 *   4. "durum" değeri: "Güvenli", "Dikkatli" veya "Tehlikeli" olmalı
 *   5. lat: -90..90, lon: -180..180 aralığında olmalı
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_FILE  (1 << 20)   /* 1 MB üst sınır */
#define MAX_KAYIT 10000

/* ── Yardımcı: karakter dizisi içinde substring ara ── */
static int icerir(const char *haystack, const char *needle) {
    return strstr(haystack, needle) != NULL;
}

/* ── Basit JSON ayrıştırma: her '{...}' bloğunu bul ── */
typedef struct {
    char blok[4096];
} Kayit;

/* ham JSON'dan kayıt bloklarını kes */
static int bloklari_cikart(const char *json, Kayit *kayitlar, int maks) {
    int sayi = 0;
    const char *p = json;
    while (*p && sayi < maks) {
        if (*p == '{') {
            /* eşleşen '}' bul */
            int derinlik = 0;
            const char *bas = p;
            while (*p) {
                if (*p == '{') derinlik++;
                else if (*p == '}') {
                    derinlik--;
                    if (derinlik == 0) { p++; break; }
                }
                p++;
            }
            int uzunluk = (int)(p - bas);
            if (uzunluk < (int)sizeof(kayitlar[sayi].blok)) {
                strncpy(kayitlar[sayi].blok, bas, uzunluk);
                kayitlar[sayi].blok[uzunluk] = '\0';
                sayi++;
            }
        } else {
            p++;
        }
    }
    return sayi;
}

/* ── "key": değerini string olarak çek ── */
static int deger_al(const char *blok, const char *anahtar, char *cikti, int boyut) {
    char ara[64];
    snprintf(ara, sizeof(ara), "\"%s\"", anahtar);
    const char *p = strstr(blok, ara);
    if (!p) return 0;
    p += strlen(ara);
    while (*p == ' ' || *p == ':' || *p == '\t') p++;
    if (*p == '"') {
        /* string değer */
        p++;
        int i = 0;
        while (*p && *p != '"' && i < boyut - 1)
            cikti[i++] = *p++;
        cikti[i] = '\0';
        return 1;
    } else {
        /* sayı değer */
        int i = 0;
        while (*p && *p != ',' && *p != '}' && *p != '\n' && i < boyut - 1)
            cikti[i++] = *p++;
        cikti[i] = '\0';
        return 1;
    }
}

/* ── Ana program ── */
int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Kullanim: %s <markers.json>\n", argv[0]);
        return 1;
    }

    /* Dosyayı oku */
    FILE *fp = fopen(argv[1], "r");
    if (!fp) {
        fprintf(stderr, "HATA: Dosya acilamadi: %s\n", argv[1]);
        return 1;
    }
    char *json = malloc(MAX_FILE);
    if (!json) { fclose(fp); fprintf(stderr, "Bellek hatasi\n"); return 1; }
    size_t okunan = fread(json, 1, MAX_FILE - 1, fp);
    json[okunan] = '\0';
    fclose(fp);

    /* Temel format kontrolü */
    const char *bas = json;
    while (*bas == ' ' || *bas == '\n' || *bas == '\r' || *bas == '\t') bas++;
    if (*bas != '[') {
        fprintf(stderr, "HATA: JSON bir dizi '[' ile baslamali.\n");
        free(json); return 1;
    }

    /* Boş dizi geçerli */
    const char *son = json + okunan - 1;
    while (son > json && (*son == ' ' || *son == '\n' || *son == '\r' || *son == '\t')) son--;
    if (*son != ']') {
        fprintf(stderr, "HATA: JSON bir dizi ']' ile bitmeli.\n");
        free(json); return 1;
    }

    /* Kayıtları çıkar */
    Kayit *kayitlar = malloc(MAX_KAYIT * sizeof(Kayit));
    if (!kayitlar) { free(json); fprintf(stderr, "Bellek hatasi\n"); return 1; }
    int toplam = bloklari_cikart(json, kayitlar, MAX_KAYIT);

    if (toplam == 0) {
        printf("Veri seti bos — dogrulama basarili (0 kayit).\n");
        free(json); free(kayitlar); return 0;
    }

    /* Zorunlu alanlar */
    const char *zorunlu[] = {"lat", "lon", "durum", "kategori", "tarih"};
    const char *gecerli_durumlar[] = {"Güvenli", "Dikkatli", "Tehlikeli"};
    int hata_sayisi = 0;

    for (int i = 0; i < toplam; i++) {
        const char *blok = kayitlar[i].blok;
        int kayit_hata = 0;

        /* Zorunlu alan kontrolü */
        for (int j = 0; j < 5; j++) {
            if (!icerir(blok, zorunlu[j])) {
                fprintf(stderr, "HATA [Kayit %d]: '%s' alani eksik.\n", i + 1, zorunlu[j]);
                kayit_hata = 1;
            }
        }

        /* Durum kontrolü */
        char durum_val[64] = {0};
        if (deger_al(blok, "durum", durum_val, sizeof(durum_val))) {
            int gecerli = 0;
            for (int j = 0; j < 3; j++) {
                if (strcmp(durum_val, gecerli_durumlar[j]) == 0) {
                    gecerli = 1; break;
                }
            }
            if (!gecerli) {
                fprintf(stderr, "HATA [Kayit %d]: Gecersiz durum degeri: '%s'\n", i + 1, durum_val);
                kayit_hata = 1;
            }
        }

        /* Koordinat aralık kontrolü */
        char lat_str[32] = {0}, lon_str[32] = {0};
        if (deger_al(blok, "lat", lat_str, sizeof(lat_str)) &&
            deger_al(blok, "lon", lon_str, sizeof(lon_str))) {
            double lat = atof(lat_str);
            double lon = atof(lon_str);
            if (lat < -90.0 || lat > 90.0) {
                fprintf(stderr, "HATA [Kayit %d]: lat degeri aralik disi: %.5f\n", i + 1, lat);
                kayit_hata = 1;
            }
            if (lon < -180.0 || lon > 180.0) {
                fprintf(stderr, "HATA [Kayit %d]: lon degeri aralik disi: %.5f\n", i + 1, lon);
                kayit_hata = 1;
            }
        }

        if (kayit_hata) hata_sayisi++;
    }

    /* Sonuç */
    if (hata_sayisi == 0) {
        printf("Dogrulama basarili: %d kayitin tamami gecerli.\n", toplam);
        free(json); free(kayitlar); return 0;
    } else {
        fprintf(stderr, "Dogrulama basarisiz: %d / %d kayit hatali.\n", hata_sayisi, toplam);
        free(json); free(kayitlar); return 1;
    }
}
