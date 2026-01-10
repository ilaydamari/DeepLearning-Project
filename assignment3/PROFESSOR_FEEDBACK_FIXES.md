# עדכונים לפי הערות המרצה - Assignment 3

## 📝 הערות המרצה ותיקונים שבוצעו

### 1. שינויים זעירים בגישות 
**הערת המרצה**: "please refrain from making only miniature changes" - הגישות לא חייבות להיות שונות לחלוטין, אבל אסור שיהיו שינויים קטנים מדי.

**✅ תיקונים שבוצעו:**

#### לפני התיקון:
- **גישה A**: Melody concatenation - שילוב מלודיה + מילים ברמת input
- **גישה B**: Melody conditioning - רק initial conditioning של hidden states

#### אחרי התיקון (שינוי משמעותי):
- **גישה A**: Direct Concatenation at Input Level
  - שילוב ישיר של melody (84D) + word embeddings (300D) = 384D input
  - עיבוד טמפורלי ישיר frame-by-frame
  - ארכיטקטורה: Combined Input → RNN → Output

- **גישה B**: Initial Conditioning + Continuous Attention (**שינוי מהותי**)
  - שלב 1: Melody → Global conditioning vector → Initial hidden states
  - שלב 2: Standard word embeddings (300D) → RNN 
  - שלב 3: **Continuous melody attention** בין RNN output למלודיה
  - שלב 4: **Gated fusion** של RNN output עם melody-attended context
  - ארכיטקטורה: Melody Conditioning → Word RNN → **Attention** → **Gated Fusion**

**ההבדלים המשמעותיים החדשים:**
- עיבוד input: A=שילוב ישיר, B=עיבוד נפרד + attention
- יישור זמני: A=ישיר frame-by-frame, B=attention גמיש  
- עומק ארכיטקטורה: A=שלב יחיד, B=רב-שלבי (4 שלבים)
- מיקום שילוב: A=ברמת input, B=initial conditioning + output attention + gating

---

### 2. בחירה דטרמיניסטית
**הערת המרצה**: "your mechanism for selecting the next word should not be deterministic (i.e., always select the word with the highest probability)"

**✅ תיקונים שבוצעו:**

#### ווידוא probabilistic sampling בכל המודלים:
```python
# ✅ נוסף לכל פונקציות הגנרציה:

# Temperature scaling (שליטה באקראיות)
if temperature != 1.0:
    next_word_logits = next_word_logits / temperature

# Top-k sampling (איזון יצירתיות-קוהרנטיות)
if top_k > 0:
    top_k_logits, top_k_indices = torch.topk(next_word_logits, top_k)
    next_word_logits = torch.full_like(next_word_logits, -float('inf'))
    next_word_logits[top_k_indices] = top_k_logits

# PROBABILISTIC sampling (never argmax)
probabilities = F.softmax(next_word_logits, dim=-1)
next_word = torch.multinomial(probabilities, num_samples=1)  # ✅ לא argmax!
```

#### הערות שנוספו בקוד:
- "PROBABILISTIC sampling (never argmax - following assignment requirements)"
- "Following professor's requirement: should not be deterministic"
- "ENSURES probabilistic sampling (never deterministic argmax)"
- בדיקה שtemperature > 0 למניעת התנהגות דטרמיניסטית

#### קבצים שעודכנו:
- ✅ `models/MelodyRNN.py` - שתי גישות למלודיה
- ✅ `models/RNN_baseline.py` - מודל בסיסי
- ✅ `models/RNN_baseline_V1.py` - LSTM
- ✅ `models/RNN_baseline_V2.py` - GRU

---

## 📋 סיכום העדכונים

### מודלים שעודכנו:
1. **MelodyConcatenationRNN (Approach A)** - ✅ probabilistic sampling
2. **MelodyConditioningRNN (Approach B)** - ✅ שינוי משמעותי + probabilistic sampling
3. **LyricsRNN (Baseline)** - ✅ probabilistic sampling מאומת
4. **V1, V2 variants** - ✅ probabilistic sampling מאומת

### README עודכן עם:
- הסבר מפורט על ההבדלים המשמעותיים בין הגישות
- הדגשת השימוש ב-probabilistic sampling
- טבלת השוואה מעודכנת
- הערות המרצה ותיקונים

### תוצאה:
הפרויקט עכשיו עומד בדרישות המרצה:
✅ **גישות שונות משמעותית** - לא שינויים קטנים
✅ **גנרציה לא-דטרמיניסטית** - רק probabilistic sampling
✅ **תיעוד מלא** של השינויים והשיפורים

---

**תאריך עדכון**: ינואר 2026  
**מותאם לדרישות**: Assignment 3 - Deep Learning Course