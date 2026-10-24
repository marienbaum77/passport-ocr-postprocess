"""Постобработка OCR паспорта РФ.

    python postprocess.py data/test_predictions.jsonl answers.jsonl

Для каждого документа решаем: accept (и отдаём 10 полей) или reject.
Только стандартная библиотека Python, CPU, без сети.
"""
import csv
import difflib
import json
import os
import re
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DATE = date(2026, 10, 1)

FIELDS = ["Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru", "Birth_date",
          "Birth_place_ru", "Issue_date", "Issue_organisation_code", "Licence_number",
          "Issue_organization_ru"]
NAME_FIELDS = ["Last_name_ru", "First_name_ru", "Middle_name_ru"]

# ---------------------------------------------------------------- MRZ

# Транслитерация внутреннего паспорта РФ в MRZ (приказ МВД): буква -> один символ.
CYR = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
LAT = "ABVGDE2JZIQKLMNOPRSTUFHC34WXY9678"
CYR2MRZ = dict(zip(CYR, LAT))
MRZ2CYR = dict(zip(LAT, CYR))

# Буквы, которые OCR путает с цифрами; применяем только в цифровых позициях
# и только если после замены сходится контрольная цифра.
DIGIT_FIX = {"O": "0", "D": "0", "Q": "0", "U": "0", "I": "1", "L": "1", "T": "7",
             "Z": "2", "S": "5", "B": "8", "G": "6", "A": "4"}


def check_digit(s):
    total = 0
    for i, c in enumerate(s):
        v = int(c) if c.isdigit() else 0 if c == "<" else ord(c) - 55
        total += v * (7, 3, 1)[i % 3]
    return str(total % 10)


def to_mrz(s):
    """Кириллица -> алфавит MRZ; пробел и дефис -> '<'; прочее -> None."""
    out = []
    for c in s:
        if c in CYR2MRZ:
            out.append(CYR2MRZ[c])
        elif c in " -":
            out.append("<")
        else:
            return None
    return "".join(out)


def digits(s):
    s = "".join(DIGIT_FIX.get(c, c) for c in s)
    return s if s.isdigit() else None


def checked(chars, check):
    """Значение поля MRZ, если оно после починки цифр сходится с контрольной цифрой."""
    v, c = digits(chars), digits(check)
    return v if v is not None and c is not None and check_digit(v) == c else None


def parse_mrz(doc):
    """Строки MRZ: line1 (имена) и поля line2, прошедшие контрольные цифры."""
    res = {"line1": None, "line1_conf": None}
    mrz = doc.get("mrz")
    if not mrz:
        return res
    line2 = None
    for ln in mrz["lines"]:
        t = ln["text"]
        if sum(c.isdigit() for c in t) >= 20:  # вторая строка почти целиком из цифр
            line2 = t
        else:
            res["line1"], res["line1_conf"] = t, ln["char_conf"]
    if not line2:
        return res
    # Левая часть строки: номер, дата рождения, пол — привязаны к началу строки.
    res["docnum"] = checked(line2[0:9], line2[9:10])
    res["birth"] = checked(line2[13:19], line2[19:20])
    # Пол контрольной цифрой не защищён; верим ему, только если строка не сдвинута.
    res["sex"] = line2[20] if res["birth"] and line2[20:21] in ("M", "F") else None
    # Правая часть: 4-я цифра серии + дата выдачи + код подразделения — привязана к концу.
    tail = line2[-16:]
    opt = checked(tail[0:13], tail[14:15]) if tail[13:14] == "<" else None
    if opt is not None:
        res["series4"], res["issue"], res["code"] = opt[0], opt[1:7], opt[7:13]
    return res


# ---------------------------------------------------------------- VIZ helpers

def viz_lines(doc, field):
    v = doc["viz"].get(field)
    return v["lines"] if v and v["lines"] else None


def viz_text(doc, field):
    lines = viz_lines(doc, field)
    return " ".join(" ".join(l["text"] for l in lines).split()) if lines else None


def geometry_ok(doc, field):
    """Строки должны заполнять рамку поля: иначе детектор потерял строку."""
    v = doc["viz"].get(field)
    fx, fy, fw, fh = v["box"]
    lines = v["lines"]
    top = min(l["box"][1] for l in lines)
    bottom = max(l["box"][1] + l["box"][3] for l in lines)
    return top - fy <= 6 and fy + fh - bottom <= 6 and fh / len(lines) < 38


# ---------------------------------------------------------------- поля

def mrz_date(yymmdd, century):
    """YYMMDD -> DD.MM.YYYY; None, если такой даты нет (значит, в MRZ >= 2 ошибок)."""
    s = f"{yymmdd[4:6]}.{yymmdd[2:4]}.{century}{yymmdd[0:2]}"
    try:
        parse_date(s)
    except ValueError:
        return None
    return s


def resolve_mrz_fields(doc, mrz):
    """Даты, номер, код — из MRZ (контрольные цифры), век года — по правилам выдачи."""
    out, bad = {}, []
    # Паспорт выдают с 14 лет, а выдача не позже 2026 г.: рождение <= 2012.
    birth = mrz.get("birth") and mrz_date(mrz["birth"], "20" if int(mrz["birth"][:2]) <= 12 else "19")
    if birth:
        out["Birth_date"] = birth
    else:
        bad.append("Birth_date")
    # Паспорта нового образца выдаются с 1997 г.
    issue = mrz.get("issue") and mrz_date(mrz["issue"], "19" if int(mrz["issue"][:2]) >= 97 else "20")
    if issue:
        out["Issue_date"] = issue
        out["Issue_organisation_code"] = mrz["code"][:3] + "-" + mrz["code"][3:]
    else:
        bad += ["Issue_date", "Issue_organisation_code"]
    if mrz.get("docnum") and mrz.get("series4"):
        n = mrz["docnum"]
        out["Licence_number"] = f"{n[0:2]} {n[2]}{mrz['series4']} {n[3:9]}"
    else:
        bad.append("Licence_number")
    return out, bad


def regime(doc):
    """Режим снимка: у mobile_v2 своя шкала уверенности; на плохих снимках (размыт, блики,
    мелкий) уверенность «шумит» — средние значения менее тревожны, высокие менее надёжны."""
    if doc["source"] == "mobile_v2":
        return "v2"
    q = doc["quality"]
    return "good" if q["sharpness"] >= 120 and q["glare"] < 0.166 and q["doc_px"] >= 1100 else "poor"


def conf_levels(doc):
    """(bad, sus, good): ниже bad символ почти всегда ошибочен, ниже sus — подозрителен,
    от good — ему можно верить. У mobile_v2 уверенность «сжата» к 1.0 (см. REPORT.md).
    """
    return (0.99, 0.999, 0.999) if doc["source"] == "mobile_v2" else (0.80, 0.95, 0.99)


# P(символ прочитан неверно) по режиму снимка и уверенности — оценка на train
# (выравнивание чтения с разметкой): [(нижняя граница уверенности, вероятность ошибки)].
VIZ_ERR = {"good": [(0.999, 0.0008), (0.98, 0.0014), (0.95, 0.0022), (0.9, 0.052), (0.8, 0.42), (0, 0.965)],
           "poor": [(0.999, 0.0027), (0.98, 0.0059), (0.95, 0.0072), (0.9, 0.0154), (0.8, 0.063), (0, 0.953)],
           "v2": [(0.999, 0.0016), (0.99, 0.10), (0, 0.97)]}
MRZ_ERR = {"good": [(0.999, 0.0009), (0.98, 0.0015), (0.95, 0.0031), (0.9, 0.0275), (0.8, 0.21), (0, 0.946)],
           "poor": [(0.999, 0.0034), (0.995, 0.0044), (0.99, 0.0059), (0.98, 0.0073), (0.95, 0.0092),
                    (0.9, 0.0151), (0.8, 0.036), (0, 0.934)],
           "v2": [(0.999, 0.0015), (0.99, 0.036), (0, 0.95)]}


def err_prob(table, doc, conf):
    return next(p for lo, p in table[regime(doc)] if conf >= lo)


def viz_chars(doc, field):
    """[(символ, уверенность)] поля; строки склеены пробелом, пробелы схлопнуты."""
    out = []
    for l in viz_lines(doc, field) or []:
        for c, x in [(" ", 1.0)] + list(zip(l["text"], l["char_conf"])):
            if c == " " and (not out or out[-1][0] == " "):
                if out:
                    out[-1] = (" ", min(out[-1][1], x))
                continue
            out.append((c, x))
    while out and out[-1][0] == " ":
        out.pop()
    return out


NAME_ODDS = 500  # символ имени правим, только если вторая версия в 500 раз вероятнее


def resolve_names(doc, mrz):
    """ФИО: VIZ сверяем с первой строкой MRZ (независимое прочтение).

    Где чтения расходятся, сравниваем шансы «ошиблась VIZ» и «ошиблась MRZ» по калибровке
    уверенности; если ни одна версия не перевешивает в NAME_ODDS раз — отказ.
    """
    bad, sus, good = conf_levels(doc)
    chars = {f: viz_chars(doc, f) for f in NAME_FIELDS}
    if not chars["Last_name_ru"] or not chars["First_name_ru"]:
        return None, "name_missing"
    line1 = mrz.get("line1")
    if not line1:
        return None, "no_mrz_line1"
    # Ожидаемая по VIZ строка 1: (символ MRZ, поле, позиция в поле); поле None — служебный.
    exp = [(c, None, None) for c in "PNRUS"]
    for k, f in enumerate(NAME_FIELDS):
        exp += [("<", None, None)] * (0, 2, 1)[k]
        for i, (c, _) in enumerate(chars[f]):
            m = to_mrz(c)
            if m is None:
                return None, "name_charset"
            exp.append((m, f, i))
    truncated = exp[44:]
    exp = (exp + [("<", None, None)] * 44)[:44]
    last = max(i for i, e in enumerate(exp) if e[1])
    obs, oc = line1, mrz["line1_conf"]
    result = {f: [c for c, _ in chars[f]] for f in NAME_FIELDS}
    inserts = []

    def vc(i):  # уверенность VIZ для элемента exp[i]
        e = exp[i] if 0 <= i < len(exp) else (None, None, None)
        return chars[e[1]][e[2]][1] if e[1] else 1.0

    def mc(j):  # уверенность MRZ для obs[j]
        return oc[j] if 0 <= j < len(oc) else 1.0

    def service(i):  # префикс PNRUS или хвост-заполнитель
        return i < 5 or i > last

    sm = difflib.SequenceMatcher(None, "".join(e[0] for e in exp), obs, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        if op == "replace" and i2 - i1 == j2 - j1:
            for i, j in zip(range(i1, i2), range(j1, j2)):
                o, co, e = obs[j], oc[j], exp[i]
                if e[1] is None:
                    if service(i) and (i < 5 or co < bad):
                        continue  # ошибка в префиксе или шум в заполнителе
                    return None, "name_mrz_mismatch"
                v, cv = chars[e[1]][e[2]]
                if v == "Ё" and o == "E":  # в разметке Ё почти не встречается
                    result[e[1]][e[2]] = "Е"
                    continue
                if o not in MRZ2CYR:  # '<' посреди имени или символ, невозможный в имени MRZ
                    if o != "<" and cv >= good:
                        continue
                    return None, "name_mrz_mismatch"
                pv = err_prob(VIZ_ERR, doc, cv)
                pm = err_prob(MRZ_ERR, doc, co)
                # Ошибка VIZ обычно даёт похожую букву, ошибка MRZ — нет.
                lr = 8.0 if MRZ2CYR[o] in CONFUSE.get(v, ()) else 0.15
                odds = pv * (1 - pm) * lr / (pm * (1 - pv))  # «ошиблась VIZ» : «ошиблась MRZ»
                if odds >= NAME_ODDS:
                    result[e[1]][e[2]] = MRZ2CYR[o]
                elif odds > 1 / NAME_ODDS:
                    return None, "name_mrz_mismatch"
        elif op == "delete":  # в VIZ есть символы, которых нет в MRZ
            for i in range(i1, i2):
                e = exp[i]
                if e[1] is None:
                    if service(i):
                        continue
                    return None, "name_mrz_mismatch"
                nb = min(mc(j1 - 1), mc(j1))
                if vc(i) < bad and nb >= sus:
                    result[e[1]][e[2]] = None  # лишний символ VIZ
                elif vc(i) >= good and nb < sus:
                    continue  # MRZ потеряла символ
                else:
                    return None, "name_mrz_mismatch"
        elif op == "insert":  # в MRZ есть символы, которых нет в VIZ
            extra = obs[j1:j2]
            ec = [mc(j) for j in range(j1, j2)]
            nb = min(vc(i1 - 1), vc(i1))
            if i1 < 5 or max(ec) < bad and nb >= sus:
                continue  # шум MRZ
            host = i1 - 1 if exp[i1 - 1][1] else i1 if i1 < len(exp) and exp[i1][1] else None
            if host is not None and min(ec) >= good and nb < sus and all(c in MRZ2CYR for c in extra):
                f, pos = exp[host][1], exp[host][2] + (host == i1 - 1)
                inserts.append((f, pos, [MRZ2CYR[c] for c in extra]))
                continue
            return None, "name_mrz_mismatch"
        else:
            return None, "name_mrz_mismatch"
    for f, pos, cs in sorted(inserts, key=lambda x: -x[1]):
        result[f][pos:pos] = cs
    # Хвост, не поместившийся в MRZ, проверить нечем — берём только очень уверенный.
    if any(chars[e[1]][e[2]][1] < max(good, 0.999) for e in truncated if e[1]):
        return None, "mrz_truncated"
    return {f: "".join(c for c in result[f] if c) for f in NAME_FIELDS}, None


PATRONYMIC_SEX = [("ВИЧ", "M"), ("ИЧ", "M"), ("ОГЛЫ", "M"), ("УЛЫ", "M"),
                  ("ВНА", "F"), ("ЧНА", "F"), ("КЫЗЫ", "F"), ("ГЫЗЫ", "F")]
SEX_RU = {"M": "МУЖ.", "F": "ЖЕН."}


def resolve_sex(doc, mrz, middle):
    """Пол из MRZ; подтверждаем текстом VIZ или окончанием отчества."""
    s = mrz.get("sex")
    if s is None:
        return None, "no_mrz_sex"
    votes = []
    t = viz_text(doc, "Sex_ru")
    if t in ("МУЖ.", "ЖЕН."):
        votes.append("M" if t == "МУЖ." else "F")
    for end, sx in PATRONYMIC_SEX:
        if middle and middle.endswith(end):
            votes.append(sx)
            break
    if not votes or any(v != s for v in votes):
        return None, "sex_conflict"
    return SEX_RU[s], None


# ---------------------------------------------------------------- место рождения, орган выдачи

ENDINGS = sorted("ОГО ЕГО ОМУ ЕМУ ЫМИ ИМИ ЫХ ИХ ИЙ ЫЙ ОЙ ЕЙ АЯ ЯЯ ОЕ ЕЕ ОМ ЕМ УЮ ЮЮ ИЕ ЫЕ "
                 "АМ ЯМ АХ ЯХ А Я Е И Ы У Ю О Ь".split(), key=len, reverse=True)


def stem(word):
    for e in ENDINGS:
        if word.endswith(e) and len(word) - len(e) >= 5:
            return word[:-len(e)]
    return word


NO_WORDS = frozenset()


class Lexicon:
    """Словарь слов места рождения и органа выдачи: справочник подразделений + train-разметка."""

    def __init__(self, texts, codes=None, places=()):
        self.count = {}
        for t in texts:
            for w in t.split():
                self.count[w] = self.count.get(w, 0) + 1
        self.place_count = {}  # только места рождения из разметки
        for t in places:
            for w in t.split():
                self.place_count[w] = self.place_count.get(w, 0) + 1
        places = list(places)
        self.lm_place = Bigrams(places)
        self.lm_org = Bigrams([t for t in texts if t not in set(places)])
        self.stems = {stem(w) for w, c in self.count.items() if c >= 2 and len(w) >= 7}
        codes_ = {}
        for code, name in codes or []:
            codes_.setdefault(code, set()).update(name.split())
        # код подразделения -> слова его названий в справочнике (frozenset: id стабилен для кэша)
        self.codes = {c: frozenset(ws) for c, ws in codes_.items()}
        self._memo = {}

    def known(self, word, extra=NO_WORDS, min_count=2):
        """Слово встречалось в словаре; «Г.РЫБИНСКЕ» — если известны «Г.» и «РЫБИНСКЕ»."""
        key = (word, id(extra), min_count)
        if key not in self._memo:
            head, dot, tail = word.partition(".")
            self._memo[key] = self.count.get(word, 0) >= min_count or word in extra or \
                bool(dot) and len(tail) >= 2 and tail[0].isalpha() and \
                self.known(head + ".", extra, min_count) and self.known(tail, extra, min_count)
        return self._memo[key]

    def plausible(self, word, extra=NO_WORDS):
        """Мягкая проверка для неисправленных слов: другая форма слова, номер, скобки."""
        w = word.strip("()\"',")
        if not w or self.known(word, extra) or self.known(w, extra) or re.fullmatch(r"№?\d+", w):
            return True
        return len(w) >= 7 and stem(w) in self.stems

    def generic(self, word):
        return self.count.get(word, 0) >= 30

    def place_word(self, word):
        """Слово, которое встречалось именно в местах рождения (а не только в названиях
        подразделений, где оно стоит в другом падеже)."""
        if self.place_count.get(word, 0) >= 2 or self.generic(word):
            return True
        head, dot, tail = word.partition(".")
        return bool(dot) and len(tail) >= 2 and self.place_word(head + ".") and self.place_word(tail)


# Пары символов, которые OCR путает (по ошибкам на train, встречались >= 3 раз).
CONFUSABLE = "ОС ВР АЛ ЖК МН ГТ ИН ЁЕ ИЙ 09 НП УЧ 08 ДЛ 27 БЬ ЛП 14 17 35 ЖХ 56 68 ШЩ ЗЭ ЦЩ ЫЬ 38 ВО ИО"
CONFUSE = {}
for _a, _b in CONFUSABLE.split():
    CONFUSE.setdefault(_a, set()).add(_b)
    CONFUSE.setdefault(_b, set()).add(_a)


def near_known(word, lex, extra):
    """Неизвестное слово в одной «типичной» замене от известного — похоже на ошибку OCR."""
    for i, c in enumerate(word):
        for a in CONFUSE.get(c, ()):
            if lex.known(word[:i] + a + word[i + 1:], extra):
                return True
    return False


class Bigrams:
    """Биграммная модель слов (Witten-Bell) — чем исправление лучше в контексте."""

    def __init__(self, texts):
        self.uni, self.bi, self.follow = {}, {}, {}
        texts = [lm_words(t) for t in texts]
        freq = {}
        for ws in texts:
            for w in ws:
                freq[w] = freq.get(w, 0) + 1
        self.vocab = {w for w, c in freq.items() if c >= 2}  # редкие слова -> <unk>
        for ws in texts:
            ws = ["<s>"] + [w if w in self.vocab else "<unk>" for w in ws] + ["</s>"]
            for v, w in zip(ws, ws[1:]):
                self.uni[w] = self.uni.get(w, 0) + 1
                self.bi[(v, w)] = self.bi.get((v, w), 0) + 1
                self.follow.setdefault(v, set()).add(w)
        self.ctx = {}
        for (v, _), c in self.bi.items():
            self.ctx[v] = self.ctx.get(v, 0) + c
        self.n = sum(self.uni.values())

    def p(self, v, w):
        v = v if v in self.vocab or v in ("<s>", "</s>") else "<unk>"
        w = w if w in self.vocab or w in ("<s>", "</s>") else "<unk>"
        pu = (self.uni.get(w, 0) + 0.5) / (self.n + 0.5 * (len(self.uni) + 1))
        t = len(self.follow.get(v, ()))
        c = self.ctx.get(v, 0)
        return (self.bi.get((v, w), 0) + t * pu) / (c + t) if c else pu

    def score_between(self, left, words, right):
        ws = [left] + words + [right]
        out = 1.0
        for v, w in zip(ws, ws[1:]):
            out *= self.p(v, w)
        return out


def lm_words(text):
    """Слова для языковой модели: склейки «Г.ПСКОВЕ» режем по точке (пробелы не моделируем)."""
    return [p for w in text.split() for p in re.split(r"(?<=\.)", w) if p]


# Модель канала OCR: что происходит с символом, если он прочитан неверно (оценка на train).
ERR_TYPES = {"sub_confusable": 0.62, "sub_other": 0.08, "inserted": 0.10, "neighbour_deleted": 0.20}
DELETED_CHAR = {".": 0.37, " ": 0.08, "-": 0.075}  # какой символ OCR чаще всего теряет


def edit_prob(c, pe, a, b, rep):
    """P(правки) по модели канала: символ c (с вероятностью ошибки pe) -> rep.
    (a, b) — заменяемый отрезок; a == b — вставка пропущенного символа рядом с c."""
    if b > a and rep == c:
        return 1 - pe
    if b > a and rep == "":
        return pe * ERR_TYPES["inserted"]
    if b > a:
        alt = CONFUSE.get(c, ())
        return pe * (ERR_TYPES["sub_confusable"] / len(alt) if rep in alt else ERR_TYPES["sub_other"] / 45)
    return pe * ERR_TYPES["neighbour_deleted"] * 0.5 * DELETED_CHAR.get(rep, 0.47 / 45)


TEXT_POSTERIOR = 0.995  # исправление принимаем, если оно настолько вероятнее альтернатив
TEXT_ALPHABET = CYR + "0123456789.-№"


def region_candidates(text, conf, s, e, bad, sus, lex, extra, is_org, doc):
    """Варианты участка text[s:e] -> P(канала): каждое подозрительное место исправлено
    одной правкой (замена, удаление, вставка рядом), все слова есть в словаре."""
    positions = [i for i in range(s, e) if conf[i] < sus]
    n_bad = sum(conf[i] < bad for i in positions)
    budget = max(1, n_bad)  # не больше одной правки сверх обязательных
    if n_bad > 2 or len(positions) > 8:
        return {}
    if not n_bad and not all(lex.known(w, extra) for w in text[s:e].split()):
        # Прочитанное слово может быть верным, но его нет в словаре (редкое название):
        # «исправить» его на известное слово было бы опасно.
        return {}
    # Слова участка как есть, включая части склеек «ПОС.ИЗНОСКИ» -> «ПОС.», «ИЗНОСКИ».
    original = {p for w in text[s:e].split() for p in [w] + re.split(r"(?<=\.)", w) if p}

    def edits(i):
        c, pe = text[i], err_prob(VIZ_ERR, doc, conf[i])
        cands = ([(i, i + 1, c)] if conf[i] >= bad else []) + [(i, i + 1, "")]  # верен / лишний
        cands += [(i, i + 1, a) for a in TEXT_ALPHABET if a != c]  # замена
        cands += [(j, j, a) for a in TEXT_ALPHABET + " " for j in (i, i + 1)]  # пропуск рядом
        return [(a, b, rep, edit_prob(c, pe, a, b, rep)) for a, b, rep in cands]

    def ok(w):
        if not lex.known(w, extra):
            return False
        if w in original:
            return True
        # Исправленное слово органа — частое или из названий этого кода;
        # исправленное слово места рождения — из мест рождения.
        if is_org:
            return all(lex.generic(p) or p in extra for p in re.split(r"(?<=\.)", w) if p)
        return lex.place_word(w)

    out = {}
    stack = [((), 0, 0)]
    while stack:
        chosen, k, changes = stack.pop()
        if k < len(positions):
            for ed in edits(positions[k]):
                change = text[ed[0]:ed[1]] != ed[2]
                if changes + change <= budget:
                    stack.append((chosen + (ed,), k + 1, changes + change))
            continue
        piece, at, prob = [], s, 1.0
        for a, b, rep, pr in sorted(chosen):
            if a < at:
                break  # правки наложились
            piece += [text[at:a], rep]
            at, prob = b, prob * pr
        else:
            words = ("".join(piece) + text[at:e]).split()
            if words and all(ok(w) for w in words):
                cand = " ".join(words)
                out[cand] = out.get(cand, 0.0) + prob
    return out


UNK_WEIGHT = 0.02  # доля «неизвестных» исходов ошибки, которые дают реальное редкое слово


def pick(cands, left, right, lm, p_err):
    """Лучший вариант участка по P(канала) * P(контекста) или None, если он не
    перевешивает остальные (включая «верное слово — редкое, его нет в словаре»)
    или варианты различаются только точками и пробелами."""
    shape = {}
    for c in cands:
        key = c.replace(" ", "").replace(".", "")
        if key in shape:
            return None  # «Г.ПСКОВЕ» / «Г. ПСКОВЕ», «ОБЛ» / «ОБЛ.» — не различаем
        shape[key] = c
    post = {c: p * lm.score_between(left, lm_words(c), right) for c, p in cands.items()}
    unk = p_err * UNK_WEIGHT * lm.score_between(left, ["<unk>"], right)
    best = max(post, key=post.get)
    return best if post[best] >= TEXT_POSTERIOR * (sum(post.values()) + unk) else None


def resolve_text(doc, field, lex, code=None):
    """Место рождения / орган выдачи: подтверждения из MRZ нет, опираемся на
    уверенность символов, словарь и контекст. Подозрительные места правим, только
    если один вариант заметно вероятнее остальных."""
    if viz_lines(doc, field) is None:
        return None, "missing"
    if not geometry_ok(doc, field):
        return None, "lost_line"
    is_org = field == "Issue_organization_ru"
    extra = lex.codes.get(code, NO_WORDS) if is_org else NO_WORDS
    lm = lex.lm_org if is_org else lex.lm_place
    bad, sus, _ = conf_levels(doc)
    chars = viz_chars(doc, field)
    text = "".join(c for c, _ in chars)
    conf = [x for _, x in chars]
    for i, c in enumerate(text):
        if c == "Ё" and is_org:  # в органе выдачи Ё не бывает: это ошибка OCR вместо Е
            conf[i] = min(conf[i], bad - 1e-6)
    # «Г.РЫБИНСКЕ» и «Г. РЫБИНСКЕ» в разметке встречаются оба. Пропущенный пробел после
    # точки OCR иногда не помечает, а лишь роняет уверенность по соседству; у mobile_v2
    # не помечает вовсе. Тогда считаем место подозрительным, чтобы рассмотреть оба варианта.
    noisy = regime(doc) != "good"
    for m in re.finditer(r"\.(?=[А-ЯЁ])", text):
        i = m.start()
        if noisy or min(conf[max(0, i - 3):i + 4]) < sus:
            conf[i + 1] = min(conf[i + 1], sus - 1e-6)
    # Так же с точкой в конце сокращения («ОБЛ» / «ОБЛ.»): на плохих снимках её потеря
    # не всегда видна по уверенности (у mobile_v2 на train такого не встретилось).
    if regime(doc) == "poor":
        for m in re.finditer(r"\S+", text):
            w, j = m.group(), m.end() - 1
            if w.endswith(".") and lex.known(w[:-1], extra) or lex.known(w + ".", extra):
                conf[j] = min(conf[j], sus - 1e-6)
    # Участки: слово с подозрительным символом (+ соседнее, если подозрителен пробел).
    regions = []
    for i, x in enumerate(conf):
        if x >= sus:
            continue
        s = text.rfind(" ", 0, i if text[i] == " " else i + 1) + 1 if i else 0
        e = text.find(" ", i + 1)
        e = len(text) if e < 0 else e
        if regions and s <= regions[-1][1]:
            regions[-1] = (regions[-1][0], max(e, regions[-1][1]))
        else:
            regions.append((s, e))
    out, at = [], 0
    for s, e in regions:
        cands = region_candidates(text, conf, s, e, bad, sus, lex, extra, is_org, doc)
        if not cands:
            return None, "unknown_word"
        left, right = lm_words(text[:s]), lm_words(text[e:])
        # Если прочитанное заведомо неверно, истинным может быть и слово не из словаря.
        p_unk = 1.0 if min(conf[s:e]) < bad and not is_org else 0.0
        best = pick(cands, left[-1] if left else "<s>", right[0] if right else "</s>", lm, p_unk)
        if best is None:
            return None, "unresolved"
        out += [text[at:s], best]
        at = e
    value = "".join(out) + text[at:]
    for w in value.split():
        if lex.plausible(w, extra):
            continue
        if is_org:
            return None, "unknown_word"  # у органа словарь почти замкнут
        if "Ё" in w:
            return None, "yo"  # Ё в разметке места рождения то есть, то нет
        if near_known(w, lex, extra):
            return None, "near_known"
    return value, None


# ---------------------------------------------------------------- действительность

# После 20 и 45 лет паспорт по Положению о паспорте действует ещё до 90 дней (пока не
# оформлен новый). Как это учтёт проверка — неизвестно, поэтому льготу не засчитываем
# (см. REPORT.md). SAFETY_DAYS — запас на разночтения «до дня рождения / включая».
GRACE_DAYS = 0
SAFETY_DAYS = 1


def parse_date(s):
    d, m, y = map(int, s.split("."))
    return date(y, m, d)


def add_years(d, n):
    try:
        return d.replace(year=d.year + n)
    except ValueError:  # 29 февраля
        return d.replace(year=d.year + n, month=3, day=1)


def passport_valid(birth, issue, on=REF_DATE):
    b, i = parse_date(birth), parse_date(issue)
    if i > on or i < add_years(b, 14):
        return False
    for age in (20, 45):
        deadline = add_years(b, age)  # день, когда паспорт подлежит замене
        if i < deadline and (on - deadline).days >= GRACE_DAYS - SAFETY_DAYS:
            return False
    return True


# ---------------------------------------------------------------- решение

def decide(doc, lex):
    """-> (ответ, список причин отказа)."""
    mrz = parse_mrz(doc)
    fields, reasons = {}, []
    names, why = resolve_names(doc, mrz)
    if names:
        fields.update(names)
    else:
        reasons.append(why)
    sex, why = resolve_sex(doc, mrz, (names or {}).get("Middle_name_ru") or viz_text(doc, "Middle_name_ru"))
    if sex:
        fields["Sex_ru"] = sex
    else:
        reasons.append(why)
    mrz_fields, bad = resolve_mrz_fields(doc, mrz)
    fields.update(mrz_fields)
    reasons += ["mrz_" + f for f in bad]
    for f in ("Birth_place_ru", "Issue_organization_ru"):
        value, why = resolve_text(doc, f, lex, fields.get("Issue_organisation_code"))
        if value is None:
            reasons.append(f"{f}:{why}")
        else:
            fields[f] = value
    if "Birth_date" in fields and "Issue_date" in fields and \
            not passport_valid(fields["Birth_date"], fields["Issue_date"]):
        reasons.append("expired")
    if reasons:
        return {"doc_id": doc["doc_id"], "decision": "reject"}, reasons
    return {"doc_id": doc["doc_id"], "decision": "accept",
            "fields": {f: fields[f] for f in FIELDS}}, []


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_subdivisions(path):
    with open(path, encoding="utf-8") as fh:
        return [(r["code"], r["name"]) for r in csv.DictReader(fh)]


def build_lexicon(subdivisions, labels):
    """subdivisions: [(код, название)], labels: [поля train-разметки]."""
    texts = [name for _, name in subdivisions]
    for g in labels:
        texts += [g["Birth_place_ru"], g["Issue_organization_ru"]]
    return Lexicon(texts, subdivisions, [g["Birth_place_ru"] for g in labels])


def load_lexicon():
    """Словарь из файлов задания: справочник подразделений и train-разметка."""
    paths = [os.path.join(HERE, "reference", "subdivisions.csv"),
             os.path.join(HERE, "data", "train_labels.jsonl")]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        sys.exit("не найдены файлы задания: " + ", ".join(missing))
    labels = [d["fields"] for d in read_jsonl(paths[1])]
    return build_lexicon(read_subdivisions(paths[0]), labels)


def main(src, dst):
    lex = load_lexicon()
    n_acc = 0
    with open(dst, "w", encoding="utf-8") as out:
        for doc in read_jsonl(src):
            try:
                ans, _ = decide(doc, lex)
            except Exception as exc:  # неожиданный вход: безопаснее отклонить
                print(f"{doc.get('doc_id')}: {exc!r}", file=sys.stderr)
                ans = {"doc_id": doc.get("doc_id"), "decision": "reject"}
            n_acc += ans["decision"] == "accept"
            out.write(json.dumps(ans, ensure_ascii=False) + "\n")
    print(f"accepted {n_acc}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
