import matplotlib
matplotlib.use('Agg')

import json, base64, io, os, re, traceback
from http.server import BaseHTTPRequestHandler
from os.path import commonprefix
from datetime import date

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from pypdf import PdfReader, PdfWriter

# ── Brand ──────────────────────────────────────────────────────────────────────
TEAL   = '#095972'
TEAL2  = '#175242'
TAN    = '#8E7E57'
TAN2   = '#C7BCA1'
CREAM  = '#FFF8F1'
CREAM2 = '#F5F2EC'
MUTED  = '#89837C'
RED    = '#B63D35'
GREEN  = '#175242'

LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'savvy_logo.png')

PAGE_W, PAGE_H = 8.5, 11.0
ML, MR = 0.07, 0.93      # left / right margin as figure fractions
CW = MR - ML             # content width


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _trunc(text, n=40):
    s = str(text)
    return s if len(s) <= n else s[:n - 1] + '…'

def _money(v, decimals=0):
    fmt = f',.{decimals}f'
    return f'-${abs(v):{fmt}}' if v < 0 else f'${v:{fmt}}'

def _mask_account(raw):
    """Mask an account number to its last 4 digits, e.g. '90123456' -> '3456'.

    Sleeved accounts (containing '_', e.g. '3177_01') keep their sleeve
    suffix visible: '3177_01' -> '3177_01' masked base -> '...' ->
    returns '3177_01' with only the base portion truncated, e.g. '7177_01'
    stays legible instead of the suffix swallowing the base digits.
    """
    raw = str(raw).strip()
    if '_' in raw:
        base, suffix = raw.split('_', 1)
        base = base.split('.')[0]
        base_masked = base[-4:] if len(base) >= 4 else base
        return f'{base_masked}_{suffix}'
    base = raw.split('.')[0]
    return base[-4:] if len(base) >= 4 else base

def _find_col(df, *candidates):
    for c in candidates:
        for col in df.columns:
            if c.lower() == col.lower() or c.lower() in col.lower():
                return col
    return None

def _strip_class_names(series):
    """Strip model prefix from asset class names.

    Orion prefixes sub-classes with the model name, e.g.:
      'Savvy Total Portfolios - US Fixed Income'
      'Savvy Total Portfolios - International'
      'Cash Equivalents'   ← left alone (no shared prefix)
      'Unassigned'         ← left alone

    We find the common prefix of the two longest names (which are always the
    model-prefixed ones) and strip it from every name that starts with it.
    Using only the top-2 avoids short generic names like 'Cash' polluting the
    commonprefix result and returning an empty string.
    """
    vals = [str(v).strip() for v in series]
    long_vals = sorted([v for v in vals if len(v) > 12], key=len, reverse=True)
    if len(long_vals) < 2:
        return series
    cp = commonprefix(long_vals[:2])
    if len(cp) <= 4:
        return series
    # Trim prefix to the last clean separator
    prefix = cp
    for sep in [' - ', ' – ']:
        idx = cp.rfind(sep)
        if idx > 3:
            prefix = cp[:idx + len(sep)]
            break
    else:
        idx = cp.rfind(' ')
        if idx > 3:
            prefix = cp[:idx + 1]
        else:
            return series
    return series.apply(
        lambda v: str(v)[len(prefix):].strip(' -').strip()
                  if str(v).startswith(prefix) else v
    )


# ── Direct Indexing proposal parsing ────────────────────────────────────────────

_MONEY_RE = re.compile(r'-?\(?\$[\d,]+(?:\.\d+)?\)?')

def _money_str_to_float(s):
    s = s.strip()
    neg = s.startswith('-') or (s.startswith('(') and s.endswith(')'))
    clean = s.replace('$', '').replace(',', '').replace('(', '').replace(')', '').lstrip('-').strip()
    v = float(clean)
    return -v if neg else v

def _find_money_after(label, text):
    """Find `label` in `text` (last occurrence) and return the first dollar
    figure that appears shortly after it."""
    idx = text.rfind(label)
    if idx == -1:
        return None
    window = text[idx + len(label): idx + len(label) + 60]
    m = _MONEY_RE.search(window)
    return _money_str_to_float(m.group()) if m else None

def _extract_di_figures(di_bytes):
    """Pull the realized-gain breakdown and tax figure from the Direct
    Indexing proposal's Tax Assessment page (Post-Transition/Recommended side).

    Returns (gl, tax, lt, st):
      gl  — Net Realized Gains/Losses (authoritative total)
      tax — Transition Tax Liability (may be None)
      lt  — Long Term portion of the realized gain (gains LT + losses LT)
      st  — Short Term portion of the realized gain (gains ST + losses ST)
    Raises ValueError if the G/L figure (the one we actually need) can't be
    located.
    """
    reader = PdfReader(io.BytesIO(di_bytes))

    target_text = None
    for page in reader.pages:
        text = page.extract_text() or ''
        if 'Net Realized Gains/Losses' in text and 'Transition Tax Liability' in text:
            target_text = text
            break

    if target_text is None:
        # Fall back to scanning the whole document
        target_text = '\n'.join((p.extract_text() or '') for p in reader.pages)

    gl  = _find_money_after('Net Realized Gains/Losses', target_text)
    tax = _find_money_after('Transition Tax Liability', target_text)

    # Long/Short Term split: look at the "Proposed Realized Gains ... Long
    # Term ... Short Term ... Proposed Realized Losses ... Long Term ...
    # Short Term ..." block that precedes "Net Realized Gains/Losses".
    lt = st = 0.0
    gains_idx = target_text.rfind('Proposed Realized Gains')
    net_idx   = target_text.find('Net Realized Gains/Losses',
                                  gains_idx if gains_idx != -1 else 0)
    if gains_idx != -1 and net_idx != -1 and net_idx > gains_idx:
        block = target_text[gains_idx:net_idx]
        vals = [_money_str_to_float(m.group()) for m in _MONEY_RE.finditer(block)]
        # Expected order: [gains_total, gains_lt, gains_st, losses_total, losses_lt, losses_st]
        gains_lt  = vals[1] if len(vals) > 1 else 0.0
        gains_st  = vals[2] if len(vals) > 2 else 0.0
        losses_lt = vals[4] if len(vals) > 4 else 0.0
        losses_st = vals[5] if len(vals) > 5 else 0.0
        lt = gains_lt + losses_lt
        st = gains_st + losses_st

    if gl is None:
        raise ValueError(
            'Could not find "Net Realized Gains/Losses" in the Direct '
            'Indexing proposal PDF — check that the Tax Assessment page is intact.'
        )
    return gl, tax, lt, st

def _merge_pdfs(main_bytes, appendix_bytes, skip_appendix_pages=()):
    """Append pages of appendix_bytes onto main_bytes, returning one PDF.

    skip_appendix_pages — 0-indexed page numbers of the appendix to omit
    (e.g. generic marketing pages not needed once the reports are combined).
    """
    writer = PdfWriter()
    for p in PdfReader(io.BytesIO(main_bytes)).pages:
        writer.add_page(p)
    for i, p in enumerate(PdfReader(io.BytesIO(appendix_bytes)).pages):
        if i in skip_appendix_pages:
            continue
        writer.add_page(p)
    out = io.BytesIO()
    writer.write(out)
    out.seek(0)
    return out.read()


# ── Drawing helpers ────────────────────────────────────────────────────────────

def _header(fig, y_top, client_name, title, logo_img):
    """Returns y after the header block."""
    h = 0.088
    ax = fig.add_axes([ML, y_top - h, CW, h])
    ax.axis('off')
    ax.text(0, 0.97, title, fontsize=20, fontweight='bold', color=TEAL,
            transform=ax.transAxes, va='top')
    ax.text(0, 0.48, f'Client: {client_name}', fontsize=9, color=MUTED,
            transform=ax.transAxes)
    report_date = date.today().strftime('%B %d, %Y')
    ax.text(0, 0.24, f'Prepared by Savvy Advisors  ·  {report_date}',
            fontsize=8, color=TAN2, transform=ax.transAxes)
    ax.axhline(y=0.02, color=TAN2, linewidth=0.8)
    if logo_img:
        lax = fig.add_axes([0.71, y_top - 0.062, 0.21, 0.054])
        lax.axis('off')
        lax.imshow(logo_img)
    return y_top - h - 0.010

def _section_hdr(fig, y, text):
    """Draws a section label with a hairline rule above; returns y after it."""
    # Hairline rule
    ax_rule = fig.add_axes([ML, y - 0.003, CW, 0.002])
    ax_rule.axis('off')
    ax_rule.axhline(y=0.5, color=TAN2, linewidth=0.6)
    # Label
    h = 0.026
    ax = fig.add_axes([ML, y - 0.003 - h, CW, h])
    ax.axis('off')
    ax.text(0, 0.35, text, fontsize=9, fontweight='bold', color=TEAL2,
            va='center', transform=ax.transAxes)
    return y - 0.003 - h - 0.008

def _table(ax, rows, col_labels, col_widths, fontsize=8.5,
           color_cols=None, right_cols=None, center_cols=None,
           hdr_color=TAN):
    """Render a styled table filling the given axes via bbox=[0,0,1,1]."""
    color_cols  = set(color_cols  or [])
    right_cols  = set(right_cols  or [])
    center_cols = set(center_cols or [])

    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   loc='upper left', cellLoc='left',
                   colWidths=col_widths, bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)

    n_data_rows = len(rows)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(hdr_color)
            cell.set_text_props(color='white', weight='bold')
            cell.get_text().set_ha('center')
            cell.set_edgecolor(hdr_color)  # blend border into fill
        else:
            cell.set_facecolor(CREAM if r % 2 == 0 else 'white')
            # Bottom separator only — no cell borders
            cell.visible_edges = 'B' if r < n_data_rows else ''
            cell.set_edgecolor('#D6CFC4')
            cell.set_linewidth(0.5)
            txt = cell.get_text().get_text()
            if c in right_cols:
                cell.get_text().set_ha('right')
            elif c in center_cols:
                cell.get_text().set_ha('center')
            if c in color_cols:
                clean = txt.replace('%','').replace(',','').replace(' ','')
                negative = clean.startswith('-') or clean.startswith('($')
                try:
                    v = float(clean.lstrip('$-(').rstrip(')'))
                    if negative:
                        v = -v
                    if v > 0:
                        cell.get_text().set_color(GREEN)
                        cell.get_text().set_weight('bold')
                    elif v < 0:
                        cell.get_text().set_color(RED)
                        cell.get_text().set_weight('bold')
                except (ValueError, TypeError):
                    pass
    return tbl

def _disclaimer(fig):
    ax = fig.add_axes([ML, 0.028, CW, 0.028])
    ax.axis('off')
    ax.axhline(y=1.0, color=TAN2, linewidth=0.5)
    ax.text(0, 0.25,
            'This analysis is for internal advisory use only. '
            'Tax estimates are approximate and do not constitute tax advice.',
            fontsize=7, color=TAN2, transform=ax.transAxes)

def _tax_panel(fig, x, y, w, rows, title, hdr_color, title_text_color='white'):
    """Draw a labeled two-column panel (label | value) for tax assessment.

    rows: list of (label, value_str, is_section, positive_bad)
          is_section  → bold label, rule below, value omitted
          positive_bad → invert color logic (positive = RED)
    Returns the bottom y of the panel.
    """
    ROW_H = _TAX_PANEL_ROW_H
    title_h = 0.024
    title_gap = 0.002
    c_h = len(rows) * ROW_H

    # Title bar — use an explicit patch so the fill survives axis('off')
    ax_t = fig.add_axes([x, y - title_h, w, title_h])
    ax_t.axis('off')
    ax_t.add_patch(mpatches.Rectangle((0, 0), 1, 1, facecolor=hdr_color,
                                       transform=ax_t.transAxes, zorder=0))
    ax_t.text(0.04, 0.44, title, fontsize=8, fontweight='bold', color=title_text_color,
              va='center', transform=ax_t.transAxes, zorder=1)

    # Content box
    ax_c = fig.add_axes([x, y - title_h - title_gap - c_h, w, c_h])
    ax_c.set_facecolor('white')
    for sp in ax_c.spines.values():
        sp.set_edgecolor(TAN2); sp.set_linewidth(0.5)
    ax_c.axis('off')
    ax_c.set_xlim(0, 1); ax_c.set_ylim(0, 1)

    n = len(rows)
    for i, row in enumerate(rows):
        lbl, val = row[0], row[1]
        is_sec   = row[2] if len(row) > 2 else False
        pos_bad  = row[3] if len(row) > 3 else False

        y_bot = 1.0 - (i + 1) / n
        y_top = 1.0 - i / n
        yf    = (y_bot + y_top) / 2

        # Alternating row background
        bg = CREAM if i % 2 == 0 else 'white'
        ax_c.add_patch(mpatches.Rectangle((0, y_bot), 1.0, 1.0 / n,
                                           facecolor=bg, linewidth=0,
                                           transform=ax_c.transAxes))

        weight = 'bold' if is_sec else 'normal'
        size   = 8.0 if is_sec else 7.5
        indent = 0.03 if is_sec else 0.09
        ax_c.text(indent, yf, lbl, fontsize=size, fontweight=weight,
                  color=TEAL2 if is_sec else '#2A2520',
                  va='center', transform=ax_c.transAxes)

        if val and val not in ('', '—'):
            try:
                clean = (val.replace('$', '').replace(',', '')
                             .replace('(', '-').replace(')', '').strip())
                fv = float(clean)
                if fv > 0:   v_color = RED if pos_bad else GREEN
                elif fv < 0: v_color = GREEN if pos_bad else RED
                else:         v_color = MUTED
            except ValueError:
                v_color = MUTED
            ax_c.text(0.97, yf, val, fontsize=size, fontweight=weight,
                      color=v_color, ha='right', va='center',
                      transform=ax_c.transAxes)
        elif val == '—':
            ax_c.text(0.97, yf, '—', fontsize=size, color=MUTED,
                      ha='right', va='center', transform=ax_c.transAxes)

        # Hairline rule below section header rows
        if is_sec and i < n - 1:
            ax_c.plot([0, 1], [y_bot, y_bot], color=TAN2, linewidth=0.4,
                      transform=ax_c.transAxes)

    return y - title_h - title_gap - c_h


# ── Row height sizing ──────────────────────────────────────────────────────────
_TAX_PANEL_ROW_H = 0.026
_TAX_PANEL_ROWS  = 7   # rows in the post-transition panel (with or without DI)

_P1_OVERHEAD = (
    0.088 + 0.010   # header + gap
  + 0.026 + 0.010   # model label + gap
  + 0.088 + 0.018   # cards + gap
  + 0.037 + 0.008   # AA section label (rule + text) + gap
  + 0.020           # gap between AA table and tax assessment
  + 0.037 + 0.008   # tax assessment section label + gap
  + 0.026           # panel title bars
  + _TAX_PANEL_ROWS * _TAX_PANEL_ROW_H   # panel content rows
  + 0.020           # gap after panels
  + 0.060           # disclaimer + bottom margin
)
_P1_AVAIL = (0.96 - 0.06) - _P1_OVERHEAD

def _row_h(n_aa_rows):
    """Row height that fills the space left for the AA table after fixed overhead."""
    return min(0.032, _P1_AVAIL / (n_aa_rows + 1))


# ── Main PDF function ──────────────────────────────────────────────────────────

def generate_pdf(excel_bytes: bytes, client_name: str, di_bytes: bytes = None,
                  di_account: str = None) -> bytes:
    # Direct Indexing proposal (optional) — parsed up front so any parsing
    # error surfaces before we spend time building the Eclipse-based pages.
    di_gl = di_tax = di_lt = di_st = None
    if di_bytes:
        di_gl, di_tax, di_lt, di_st = _extract_di_figures(di_bytes)

    xlsx = pd.ExcelFile(io.BytesIO(excel_bytes))

    required = {
        'Model Tolerance':           'Model Tolerance',
        'Holding and Trade Details': 'Holding and Trade Details',
        'Gain Loss Details':         'Gain Loss Details',
        'Account and Cash Details':  'Account and Cash Details',
    }
    dfs = {}
    for key, name in required.items():
        found = next((s for s in xlsx.sheet_names if name.lower() in s.lower()), None)
        if not found:
            sheets = ', '.join(f'"{s}"' for s in xlsx.sheet_names)
            raise ValueError(f'Could not find sheet "{name}". Sheets in file: {sheets}')
        dfs[key] = pd.read_excel(xlsx, sheet_name=found)

    mt_df = dfs['Model Tolerance']
    ht_df = dfs['Holding and Trade Details']
    gl_df = dfs['Gain Loss Details']
    ac_df = dfs['Account and Cash Details']

    # Load Rebalance Summary (optional — not present in all exports)
    rb_data = {}
    rb_found = next((s for s in xlsx.sheet_names if 'rebalance summary' in s.lower()), None)
    if rb_found:
        try:
            rb_df = pd.read_excel(xlsx, sheet_name=rb_found)
            if not rb_df.empty:
                rb_row = rb_df.iloc[0]
                if '# of Trades' in rb_df.columns and not pd.isna(rb_row['# of Trades']):
                    rb_data['# of Trades'] = rb_row['# of Trades']
        except Exception:
            pass

    # Target model — derive the base model name from all unique Model Category values.
    # e.g. ["STP - All Fixed - Fixed Income", "STP - All Fixed - Cash"] → "STP - All Fixed"
    # e.g. ["Savvy Strategic 70/30 Equity", "Savvy Strategic 70/30 Fixed Income"] → "Savvy Strategic 70/30"
    _skip = {'unassigned', 'nan', 'none', 'cash', ''}
    candidates = sorted({str(m).strip() for m in ht_df['Model Category'].dropna()
                         if str(m).strip().lower() not in _skip})
    if not candidates:
        target_model = 'Not specified'
    elif len(candidates) == 1:
        target_model = candidates[0]
    else:
        cp = commonprefix(candidates)
        trimmed = None
        for sep in [' - ', ' ']:
            idx = cp.rfind(sep)
            if idx > 3:
                trimmed = cp[:idx].strip()
                break
        target_model = trimmed if trimmed else candidates[0]

    # Asset allocation — strip common prefix so "Savvy Strategic 60/40 US Equity"
    # becomes "US Equity" regardless of which model is in the file.
    aa = mt_df[['Class', 'Current %', 'Target %', 'Post %', 'Trade $']].copy()
    aa['Class'] = aa['Class'].str.strip()
    aa['Class'] = _strip_class_names(aa['Class'])
    aa['Class'] = aa['Class'].str.replace('U.S.', 'US', regex=False).str.strip()
    aa['Class'] = aa['Class'].str.replace('Unassigned', 'Non-Model Holdings', case=False, regex=False)
    aa = aa.sort_values('Target %', ascending=False).reset_index(drop=True)

    # Trades
    trades = ht_df[ht_df['Trade $'].notna() & (ht_df['Trade $'] != 0)].copy()
    trades = trades[~trades['Ticker'].astype(str).str.contains(
        'CUSTODIAL_CASH', case=False, na=False)]
    trades['abs_trade'] = trades['Trade $'].abs()
    trades['Acct4'] = trades['Account Number'].apply(_mask_account)
    if 'Trade G/L $' not in trades.columns:
        trades['Trade G/L $'] = 0.0
    trades['Trade G/L $'] = trades['Trade G/L $'].fillna(0)

    buys  = (trades[trades['Trade $'] > 0]
             .sort_values('abs_trade', ascending=False).head(10)
             .reset_index(drop=True))
    sells = (trades[trades['Trade $'] < 0]
             .sort_values('abs_trade', ascending=False).head(10)
             .reset_index(drop=True))

    # Financials
    if gl_df.empty:
        raise ValueError('Gain Loss Details sheet is empty.')
    cg        = gl_df.iloc[0]
    total_val = ac_df['Account Value'].sum()

    # ── Direct Indexing account: fold its balance into the Asset Allocation
    # table (as "US Large Cap") and rebase every row's percentages against
    # the combined total, since the DI sleeve isn't part of the Model
    # Tolerance sheet at all.
    di_amount = None
    if di_account:
        acct_num_col = _find_col(ac_df, 'Account Number', 'Account Num', 'Acct Number')
        val_col      = _find_col(ac_df, 'Account Value', 'Acct Value', 'Market Value')
        if acct_num_col and val_col:
            match = ac_df[ac_df[acct_num_col].astype(str).str.strip() == str(di_account).strip()]
            if not match.empty:
                di_amount = float(match[val_col].sum())
        if di_amount is None:
            raise ValueError(
                f'Could not find account "{di_account}" in the Account and '
                'Cash Details sheet to compute the Direct Indexing allocation.'
            )

    if di_amount:
        grand_total = total_val                 # already includes the DI account
        core_total  = grand_total - di_amount    # base the Model Tolerance % were computed against

        aa['_cur_$']   = aa['Current %'].fillna(0) / 100 * core_total
        aa['_tgt_$']   = aa['Target %'].fillna(0)  / 100 * core_total
        aa['_post_$']  = aa['Post %'].fillna(0)    / 100 * core_total
        aa['_trade_$'] = aa['Trade $'].fillna(0)

        mask = aa['Class'].str.strip().str.lower() == 'us large cap'
        if mask.any():
            aa.loc[mask, '_cur_$']   += di_amount
            aa.loc[mask, '_tgt_$']   += di_amount
            aa.loc[mask, '_post_$']  += di_amount
            aa.loc[mask, '_trade_$'] += di_amount
        else:
            new_row = pd.DataFrame([{
                'Class': 'US Large Cap', 'Current %': None, 'Target %': None,
                'Post %': None, 'Trade $': None,
                '_cur_$': di_amount, '_tgt_$': di_amount,
                '_post_$': di_amount, '_trade_$': di_amount,
            }])
            aa = pd.concat([aa, new_row], ignore_index=True)

        aa['Current %'] = aa['_cur_$']  / grand_total * 100
        aa['Target %']  = aa['_tgt_$']  / grand_total * 100
        aa['Post %']    = aa['_post_$'] / grand_total * 100
        aa['Trade $']   = aa['_trade_$']
        aa = aa.drop(columns=['_cur_$', '_tgt_$', '_post_$', '_trade_$'])
        aa = aa.sort_values('Target %', ascending=False).reset_index(drop=True)

    # Logo
    logo_img = None
    try:
        logo_img = Image.open(LOGO_PATH)
    except Exception:
        pass

    # Adaptive row height for the AA table
    rh = _row_h(len(aa))

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 1 — Portfolio Summary
        # ══════════════════════════════════════════════════════════════════════
        fig = plt.figure(figsize=(PAGE_W, PAGE_H), dpi=200)
        fig.patch.set_facecolor(CREAM2)
        y = 0.96

        y = _header(fig, y, client_name, 'Portfolio Transition Analysis', logo_img)

        # Target model row
        ax_tm = fig.add_axes([ML, y - 0.024, CW, 0.022])
        ax_tm.axis('off')
        ax_tm.text(0, 0.5, f'Target Model:  {target_model}',
                   fontsize=9, color=MUTED, va='center', transform=ax_tm.transAxes)
        y -= 0.026 + 0.010

        # ── Metrics cards ─────────────────────────────────────────────────────
        gl_val   = cg['Trade Total Gain $']
        est_tax  = cg['Estimated Tax']
        if di_gl is not None:
            gl_val = gl_val + di_gl
        if di_tax is not None:
            est_tax = est_tax + di_tax
        tax_pct = (est_tax / total_val * 100) if total_val else 0
        def _signed_color(v, positive_bad=False):
            if v > 0:  return RED if positive_bad else GREEN
            if v < 0:  return GREEN if positive_bad else RED
            return MUTED
        metrics = [
            ('TOTAL VALUE',    _money(total_val),    TEAL),
            ('TRANSITION G/L', _money(gl_val, 2),    _signed_color(gl_val)),
            ('ESTIMATED TAX',  _money(est_tax, 2),   _signed_color(est_tax, positive_bad=True)),
            ('TAX IMPACT',     f'{tax_pct:.2f}%',    _signed_color(est_tax, positive_bad=True)),
        ]
        card_h = 0.088
        n_cards = len(metrics)
        gap = 0.012
        card_w = (CW - gap * (n_cards - 1)) / n_cards
        for i, (label, value, vcolor) in enumerate(metrics):
            x = ML + i * (card_w + gap)
            ax_c = fig.add_axes([x, y - card_h, card_w, card_h])
            ax_c.set_facecolor('white')
            ax_c.set_xticks([]); ax_c.set_yticks([])
            for sp in ax_c.spines.values():
                sp.set_edgecolor(TAN2); sp.set_linewidth(0.6)
            ax_c.text(0.10, 0.80, label, fontsize=7, fontweight='bold',
                      color=MUTED, transform=ax_c.transAxes)
            ax_c.text(0.10, 0.26, value, fontsize=13, fontweight='bold',
                      color=vcolor, transform=ax_c.transAxes)
        y -= card_h + 0.018

        # ── Asset allocation — full-width table ────────────────────────────────
        y = _section_hdr(fig, y, 'ASSET ALLOCATION')

        n_aa  = len(aa)
        tbl_h = (n_aa + 1) * rh

        ax_aa = fig.add_axes([ML, y - tbl_h, CW, tbl_h])
        ax_aa.axis('off')
        aa_rows = []
        for _, r in aa.iterrows():
            t = r['Trade $']
            aa_rows.append([
                _trunc(r['Class'], 38),
                f"{r['Current %']:.1f}%",
                f"{r['Target %']:.1f}%",
                f"{r['Post %']:.1f}%",
                '—' if (pd.isna(t) or t == 0) else _money(t),
            ])
        _table(ax_aa, aa_rows,
               col_labels=['Asset Class', 'Current %', 'Target %', 'Post-Trade %', 'Trade Amount'],
               col_widths=[0.38, 0.13, 0.13, 0.15, 0.21],
               fontsize=8.5,
               color_cols=[3, 4], right_cols=[4], center_cols=[1, 2, 3])
        y -= tbl_h + 0.020

        # ── Tax Assessment — pre/post panels ──────────────────────────────────
        y = _section_hdr(fig, y, 'TAX ASSESSMENT')

        # Helper: safely read a value from gl_df row (flexible column matching)
        def _cg_get(*keys):
            for k in keys:
                if k in cg.index and not pd.isna(cg[k]):
                    return float(cg[k])
            for k in keys:
                kl = k.lower()
                for col in cg.index:
                    if kl in col.lower() and not pd.isna(cg[col]):
                        return float(cg[col])
            return None

        def _fv(v, decimals=2):
            return '—' if v is None else _money(v, decimals)

        # Known Orion Gain Loss Details columns
        post_lt      = _cg_get('Trade Long Term Gain')
        post_st      = _cg_get('Trade Short Term Gain')
        post_total   = _cg_get('Trade Total Gain $')
        post_ytd     = _cg_get('Post-Trade YTD Gain')
        post_tax     = _cg_get('Estimated Tax')

        # Aggregate unrealized G/L from holdings (split gains vs losses)
        unrlzd = ht_df['Unrealized G/L $'].dropna() if 'Unrealized G/L $' in ht_df.columns else pd.Series(dtype=float)
        unrlzd_gains  = unrlzd[unrlzd > 0].sum() if not unrlzd.empty else None
        unrlzd_losses = unrlzd[unrlzd < 0].sum() if not unrlzd.empty else None
        unrlzd_net    = unrlzd.sum() if not unrlzd.empty else None
        if unrlzd_gains == 0:  unrlzd_gains  = None
        if unrlzd_losses == 0: unrlzd_losses = None
        if unrlzd_net == 0:    unrlzd_net    = None

        n_trades_str = (str(int(rb_data['# of Trades']))
                        if '# of Trades' in rb_data else '—')
        n_trades_label = '# of Trades (Not Including DI Rebalance)' if di_gl is not None else '# of Trades'

        pre_rows = [
            ('Unrealized G/L',  _fv(unrlzd_net),    False, False),
            ('  Gains',         _fv(unrlzd_gains),  False, False),
            ('  Losses',        _fv(unrlzd_losses), False, True),
        ]

        has_di = di_gl is not None
        combined_lt    = (post_lt  or 0) + (di_lt or 0) if has_di else post_lt
        combined_st    = (post_st  or 0) + (di_st or 0) if has_di else post_st
        combined_total = (post_total or 0) + di_gl       if has_di else post_total
        combined_ytd   = (post_ytd or 0) + di_gl         if has_di else post_ytd
        combined_tax   = (post_tax  or 0) + (di_tax or 0) if has_di else post_tax

        post_rows = [
            ('Realized Gains',        '',               True,  False),
            ('  Long Term',           _fv(combined_lt), False, False),
            ('  Short Term',          _fv(combined_st), False, False),
            ('Net Realized G/L',      _fv(combined_total),  False, False),
            ('YTD Gain (Post-Trade)', _fv(combined_ytd), False, False),
            ('Estimated Tax',         _fv(combined_tax),    False, True),
            (n_trades_label,          n_trades_str,     False, False),
        ]

        panel_w = (CW - 0.014) / 2
        pre_y  = _tax_panel(fig, ML,                    y, panel_w, pre_rows,
                             'PRE-TRANSITION',  '#6B6B6B')
        post_y = _tax_panel(fig, ML + panel_w + 0.014, y, panel_w, post_rows,
                             'POST-TRANSITION', '#4A4A4A')
        y = min(pre_y, post_y)

        _disclaimer(fig)
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 2 — Trade Detail
        # ══════════════════════════════════════════════════════════════════════
        fig2 = plt.figure(figsize=(PAGE_W, PAGE_H), dpi=200)
        fig2.patch.set_facecolor(CREAM2)
        y2 = 0.96

        y2 = _header(fig2, y2, client_name, 'Detailed Trade Analysis', logo_img)
        y2 -= 0.005

        # ── Account breakdown ──────────────────────────────────────────────────
        acct_num_col   = _find_col(ac_df, 'Account Number', 'Account Num', 'Acct Number')
        reg_type_col   = _find_col(ac_df, 'Reg Type', 'Registration Type', 'Reg. Type')
        acct_val_col   = _find_col(ac_df, 'Account Value', 'Acct Value', 'Market Value')
        acct_trade_col = _find_col(ac_df, 'Trade $', 'Trade Amount')

        if acct_num_col and acct_val_col:
            ac_show = ac_df[ac_df[acct_val_col].notna() & (ac_df[acct_val_col] != 0)].copy()
            if not ac_show.empty:
                acct_rows_p2 = []
                for _, r in ac_show.iterrows():
                    acct_str = f'x{_mask_account(r[acct_num_col])}'
                    reg  = str(r[reg_type_col]).strip() if reg_type_col else '—'
                    val  = _money(r[acct_val_col]) if not pd.isna(r[acct_val_col]) else '—'
                    acct_rows_p2.append([acct_str, reg, val])

                y2 = _section_hdr(fig2, y2, 'ACCOUNTS')
                n_a = len(acct_rows_p2)
                a_rh = min(0.032, 0.22 / (n_a + 1))
                a_h  = (n_a + 1) * a_rh
                ax_a = fig2.add_axes([ML, y2 - a_h, CW, a_h])
                ax_a.axis('off')
                _table(ax_a, acct_rows_p2,
                       col_labels=['Account', 'Reg Type', 'Account Value'],
                       col_widths=[0.18, 0.27, 0.55],
                       fontsize=8.5,
                       right_cols=[2],
                       center_cols=[0])
                y2 -= a_h + 0.022

        # Row height: fit the larger table into half the available space, cap at 0.032
        max_rows = max(len(buys), len(sells), 1)
        trade_overhead = 2 * (0.034 + 0.008 + 0.028)
        trade_space = y2 - 0.088 - trade_overhead
        trade_rh = max(0.022, min(0.032, trade_space / (2 * (max_rows + 1))))

        for data, title, hcol in [(buys, 'TOP 10 BUYS', TEAL),
                                   (sells, 'TOP 10 SELLS', RED)]:
            # Section label
            ax_lbl = fig2.add_axes([ML, y2 - 0.026, CW, 0.024])
            ax_lbl.axis('off')
            ax_lbl.text(0, 0.35, title, fontsize=10, fontweight='bold',
                        color=hcol, va='center', transform=ax_lbl.transAxes)
            y2 -= 0.026 + 0.008

            if data.empty:
                ax_e = fig2.add_axes([ML, y2 - 0.040, CW, 0.035])
                ax_e.axis('off')
                ax_e.text(0.5, 0.5, 'No trades identified.', ha='center',
                          fontsize=9, color=MUTED, transform=ax_e.transAxes)
                y2 -= 0.050
                continue

            is_sell = (hcol == RED)
            n_rows  = len(data) + 1  # including header
            t_h     = n_rows * trade_rh

            if is_sell:
                d_rows = [
                    [f'x{r["Acct4"]}',
                     str(r['Ticker']).lstrip('-'),
                     _trunc(r['Security Name'], 36),
                     _money(abs(r['Trade $'])),
                     _money(r['Trade G/L $'], 2)]
                    for _, r in data.iterrows()
                ]
                col_labels = ['Account', 'Ticker', 'Security Name', 'Trade $', 'G/L $']
                col_widths  = [0.13, 0.09, 0.40, 0.19, 0.19]
                right_cols  = [3, 4]
                color_cols  = [4]
            else:
                d_rows = [
                    [f'x{r["Acct4"]}',
                     str(r['Ticker']),
                     _trunc(r['Security Name'], 42),
                     _money(abs(r['Trade $']))]
                    for _, r in data.iterrows()
                ]
                col_labels = ['Account', 'Ticker', 'Security Name', 'Trade $']
                col_widths  = [0.14, 0.10, 0.52, 0.24]
                right_cols  = [3]
                color_cols  = []

            ax_t = fig2.add_axes([ML, y2 - t_h, CW, t_h])
            ax_t.axis('off')
            _table(ax_t, d_rows,
                   col_labels=col_labels,
                   col_widths=col_widths,
                   fontsize=8.5,
                   color_cols=color_cols,
                   right_cols=right_cols,
                   center_cols=[0, 1],
                   hdr_color=hcol)
            y2 -= t_h + 0.028

        # ── Direct Indexing Rebalance (kept separate from buys/sells) ──────────
        if di_gl is not None:
            ax_lbl = fig2.add_axes([ML, y2 - 0.026, CW, 0.024])
            ax_lbl.axis('off')
            ax_lbl.text(0, 0.35, 'DIRECT INDEXING REBALANCE', fontsize=10, fontweight='bold',
                        color=TEAL2, va='center', transform=ax_lbl.transAxes)
            y2 -= 0.026 + 0.008

            di_row = [['DI', 'Direct Indexing Rebalance — See Transition Analysis', _money(di_gl, 2)]]
            di_h = 2 * trade_rh
            ax_di = fig2.add_axes([ML, y2 - di_h, CW, di_h])
            ax_di.axis('off')
            _table(ax_di, di_row,
                   col_labels=['Ticker', 'Description', 'Realized G/L'],
                   col_widths=[0.12, 0.63, 0.25],
                   fontsize=8.5,
                   color_cols=[2],
                   right_cols=[2],
                   center_cols=[0],
                   hdr_color=TAN)
            y2 -= di_h + 0.028

        _disclaimer(fig2)
        pdf.savefig(fig2, facecolor=fig2.get_facecolor())
        plt.close(fig2)

    buf.seek(0)
    main_pdf_bytes = buf.read()

    if di_bytes:
        return _merge_pdfs(main_pdf_bytes, di_bytes, skip_appendix_pages={1, 2, 3})
    return main_pdf_bytes


# ── Vercel handler ─────────────────────────────────────────────────────────────

class handler(BaseHTTPRequestHandler):

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            body   = json.loads(self.rfile.read(length))

            client_name = body.get('client_name', '').strip()
            excel_b64   = body.get('excel_data', '')
            di_b64      = body.get('di_data', '')
            di_account  = body.get('di_account', '').strip() or None

            if not client_name or not excel_b64:
                self._error(400, 'client_name and excel_data are required.')
                return

            di_bytes = base64.b64decode(di_b64) if di_b64 else None
            pdf_bytes = generate_pdf(base64.b64decode(excel_b64), client_name, di_bytes, di_account)

            self.send_response(200)
            self.send_header('Content-Type', 'application/pdf')
            self.send_header('Content-Disposition',
                             'attachment; filename="Savvy_Transition.pdf"')
            self.send_header('Content-Length', str(len(pdf_bytes)))
            self.end_headers()
            self.wfile.write(pdf_bytes)

        except ValueError as e:
            self._error(422, str(e))
        except Exception:
            self._error(500, traceback.format_exc())

    def _error(self, code, message):
        body = json.dumps({'error': message}).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass
