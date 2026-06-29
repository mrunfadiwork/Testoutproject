"""
app.py - Sistem Absensi Berbasis Pengenalan Wajah
Main entry point dengan GUI Tkinter modern.
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import threading
import queue
import cv2
import numpy as np
from PIL import Image, ImageTk
from datetime import datetime
import os

import database as db
from enrollment import EnrollmentSession
from attendance import AttendanceSession
from config import *

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_dark_style(root):
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".",            background=THEME_BG,    foreground=THEME_TEXT,
                    font=("Segoe UI", 10))
    style.configure("TFrame",       background=THEME_BG)
    style.configure("TLabel",       background=THEME_BG,    foreground=THEME_TEXT)
    style.configure("TLabelframe",  background=THEME_PANEL, foreground=THEME_TEXT,
                    relief="flat", borderwidth=1)
    style.configure("TLabelframe.Label", background=THEME_PANEL, foreground=THEME_ACCENT,
                    font=("Segoe UI Semibold", 10))
    style.configure("TNotebook",    background=THEME_BG,    borderwidth=0)
    style.configure("TNotebook.Tab", background=THEME_CARD, foreground=THEME_MUTED,
                    padding=[16,8], font=("Segoe UI", 10))
    style.map("TNotebook.Tab",
              background=[("selected", THEME_PANEL)],
              foreground=[("selected", THEME_ACCENT)])
    style.configure("Treeview", background=THEME_CARD, foreground=THEME_TEXT,
                    fieldbackground=THEME_CARD, rowheight=28, borderwidth=0)
    style.configure("Treeview.Heading", background=THEME_PANEL,
                    foreground=THEME_ACCENT, font=("Segoe UI Semibold", 10))
    style.map("Treeview", background=[("selected", THEME_ACCENT)])
    style.configure("TScrollbar", background=THEME_PANEL, troughcolor=THEME_BG,
                    borderwidth=0, arrowcolor=THEME_MUTED)


def btn(parent, text, cmd, color=None, width=18):
    c = color or THEME_ACCENT
    b = tk.Button(parent, text=text, command=cmd, bg=c, fg="white",
                  font=("Segoe UI Semibold", 10), relief="flat",
                  activebackground=THEME_CARD, activeforeground=THEME_TEXT,
                  cursor="hand2", width=width, pady=6)
    b.bind("<Enter>", lambda e: b.config(bg=THEME_CARD))
    b.bind("<Leave>", lambda e: b.config(bg=c))
    return b


def card(parent, title="", **kwargs):
    f = ttk.LabelFrame(parent, text=title, **kwargs)
    f.configure(padding=10)
    return f


# ─────────────────────────────────────────────────────────────────────────────
# Enrollment Tab
# ─────────────────────────────────────────────────────────────────────────────

class EnrollmentTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.session   = None
        self.thread    = None
        self.q         = queue.Queue()
        self._build()
        self._poll()

    def _build(self):
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)

        # ── Left: camera feed ──
        left = card(self, "📷 Kamera Enrollment", padding=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(10,5), pady=10)

        self.cam_label = tk.Label(left, bg="black", width=54, height=22)
        self.cam_label.pack(fill="both", expand=True, pady=(0,6))

        self.prog_var = tk.IntVar()
        self.prog_bar = ttk.Progressbar(left, variable=self.prog_var,
                                        maximum=ENROLL_MAX_FRAMES, length=300)
        self.prog_bar.pack(fill="x")

        self.frame_lbl = tk.Label(left, text="0 frame valid terkumpul",
                                  bg=THEME_PANEL, fg=THEME_MUTED,
                                  font=("Segoe UI", 9))
        self.frame_lbl.pack(pady=3)

        # ── Right: controls + log ──
        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=(5,10), pady=10)
        right.rowconfigure(1, weight=1)

        ctrl = card(right, "⚙️ Kontrol")
        ctrl.grid(row=0, column=0, sticky="ew", pady=(0,8))

        tk.Label(ctrl, text="Nama Peserta:", bg=THEME_PANEL,
                 fg=THEME_TEXT, font=("Segoe UI", 10)).grid(row=0, column=0,
                 sticky="w", padx=5, pady=4)
        self.name_var = tk.StringVar()
        name_entry = tk.Entry(ctrl, textvariable=self.name_var,
                              bg=THEME_CARD, fg=THEME_TEXT,
                              insertbackground=THEME_TEXT,
                              font=("Segoe UI", 11), relief="flat",
                              width=22, bd=4)
        name_entry.grid(row=0, column=1, padx=5, pady=4)

        info = tk.Label(ctrl, bg=THEME_PANEL, fg=THEME_MUTED,
                        font=("Segoe UI", 8), justify="left",
                        text="Gerakan:\n• Hadap lurus\n• Kanan ↔ Kiri pelan\n• Atas ↕ Bawah pelan")
        info.grid(row=1, column=0, columnspan=2, sticky="w", padx=5)

        bf = tk.Frame(ctrl, bg=THEME_PANEL)
        bf.grid(row=2, column=0, columnspan=2, pady=8)
        self.start_btn = btn(bf, "▶  Mulai Rekam", self._start, THEME_GREEN)
        self.start_btn.pack(side="left", padx=4)
        self.stop_btn  = btn(bf, "⏹  Stop", self._stop, THEME_RED)
        self.stop_btn.pack(side="left", padx=4)
        self.stop_btn.config(state="disabled")

        self.status_lbl = tk.Label(ctrl, text="Siap", bg=THEME_PANEL,
                                   fg=THEME_ACCENT, font=("Segoe UI", 9),
                                   wraplength=220, justify="left")
        self.status_lbl.grid(row=3, column=0, columnspan=2, padx=5, pady=4)

        log_f = card(right, "📋 Log")
        log_f.grid(row=1, column=0, sticky="nsew")
        right.rowconfigure(1, weight=1)

        self.log_box = tk.Text(log_f, bg=THEME_CARD, fg=THEME_TEXT,
                               font=("Consolas", 9), relief="flat",
                               state="disabled", wrap="word", height=14)
        sc = ttk.Scrollbar(log_f, command=self.log_box.yview)
        self.log_box.config(yscrollcommand=sc.set)
        self.log_box.pack(side="left", fill="both", expand=True)
        sc.pack(side="right", fill="y")

    # ── Logic ─────────────────────────────────────────────────────────────────

    def _start(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("Input Kosong", "Masukkan nama peserta terlebih dahulu.")
            return
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.prog_var.set(0)
        self._log_append(f"\n─── Enrollment: {name} ───\n")

        self.session = EnrollmentSession(
            user_name          = name,
            on_frame_callback  = lambda f, g, s: self.q.put(("frame", f, g, s)),
            on_status_callback = lambda s: self.q.put(("log", s)),
            on_complete_callback = lambda ok, msg: self.q.put(("done", ok, msg)),
        )
        self.thread = threading.Thread(target=self.session.start, daemon=True)
        self.thread.start()

    def _stop(self):
        if self.session:
            self.session.stop()

    def _poll(self):
        try:
            while True:
                item = self.q.get_nowait()
                if item[0] == "frame":
                    _, frame, good, status = item
                    self._show_frame(frame)
                    self.prog_var.set(good)
                    self.frame_lbl.config(text=f"{good} frame valid terkumpul")
                    self.status_lbl.config(text=status)
                elif item[0] == "log":
                    self._log_append(item[1] + "\n")
                elif item[0] == "done":
                    _, ok, msg = item
                    self._on_done(ok, msg)
        except queue.Empty:
            pass
        self.after(30, self._poll)

    def _show_frame(self, frame_bgr):
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (480, 360))
        photo = ImageTk.PhotoImage(Image.fromarray(img))
        self.cam_label.config(image=photo)
        self.cam_label.image = photo

    def _on_done(self, ok, msg):
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.cam_label.config(image="")
        self._log_append(msg + "\n")
        self.status_lbl.config(text="Selesai" if ok else "Gagal",
                               fg=THEME_GREEN if ok else THEME_RED)
        if ok:
            messagebox.showinfo("Enrollment Berhasil", msg)
        else:
            messagebox.showerror("Enrollment Gagal", msg)

    def _log_append(self, text):
        self.log_box.config(state="normal")
        self.log_box.insert("end", text)
        self.log_box.see("end")
        self.log_box.config(state="disabled")


# ─────────────────────────────────────────────────────────────────────────────
# Attendance Tab
# ─────────────────────────────────────────────────────────────────────────────

class AttendanceTab(ttk.Frame):
    def __init__(self, master, on_refresh):
        super().__init__(master)
        self.on_refresh = on_refresh
        self.session    = None
        self.thread     = None
        self.q          = queue.Queue()
        self._build()
        self._poll()

    def _build(self):
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)

        # ── Left: camera ──
        left = card(self, "📷 Kamera Absensi", padding=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(10,5), pady=10)

        self.cam_label = tk.Label(left, bg="black", width=54, height=22)
        self.cam_label.pack(fill="both", expand=True, pady=(0,6))

        self.status_lbl = tk.Label(left, text="Kamera tidak aktif",
                                   bg=THEME_PANEL, fg=THEME_MUTED,
                                   font=("Segoe UI", 9))
        self.status_lbl.pack()

        bf = tk.Frame(left, bg=THEME_PANEL)
        bf.pack(pady=6)
        self.start_btn = btn(bf, "▶  Mulai Absensi", self._start, THEME_GREEN)
        self.start_btn.pack(side="left", padx=4)
        self.stop_btn  = btn(bf, "⏹  Stop Absensi", self._stop, THEME_RED)
        self.stop_btn.pack(side="left", padx=4)
        self.stop_btn.config(state="disabled")

        # ── Right: live log ──
        right = card(self, "📋 Log Absensi Hari Ini", padding=10)
        right.grid(row=0, column=1, sticky="nsew", padx=(5,10), pady=10)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        cols = ("Nama", "Waktu", "Similarity")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=18)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=120 if c != "Nama" else 160)
        sc = ttk.Scrollbar(right, command=self.tree.yview)
        self.tree.config(yscrollcommand=sc.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        sc.grid(row=0, column=1, sticky="ns")

        self._refresh_tree()

    # ── Logic ─────────────────────────────────────────────────────────────────

    def _start(self):
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.session = AttendanceSession(
            on_frame_callback      = lambda f: self.q.put(("frame", f)),
            on_attendance_callback = lambda n, s, t: self.q.put(("att", n, s, t)),
            on_status_callback     = lambda m: self.q.put(("log", m)),
            on_stop_callback       = lambda: self.q.put(("stop",)),
        )
        self.thread = threading.Thread(target=self.session.start, daemon=True)
        self.thread.start()

    def _stop(self):
        if self.session:
            self.session.stop()

    def _poll(self):
        try:
            while True:
                item = self.q.get_nowait()
                if item[0] == "frame":
                    self._show_frame(item[1])
                elif item[0] == "att":
                    _, name, sim, ts = item
                    self.tree.insert("", 0, values=(
                        name,
                        ts.strftime("%H:%M:%S"),
                        f"{sim:.3f}"
                    ))
                    self.on_refresh()
                elif item[0] == "log":
                    self.status_lbl.config(text=item[1])
                elif item[0] == "stop":
                    self._on_stopped()
        except queue.Empty:
            pass
        self.after(30, self._poll)

    def _show_frame(self, frame_bgr):
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (480, 360))
        photo = ImageTk.PhotoImage(Image.fromarray(img))
        self.cam_label.config(image=photo)
        self.cam_label.image = photo

    def _on_stopped(self):
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.cam_label.config(image="")
        self.status_lbl.config(text="Kamera dihentikan.")
        self._refresh_tree()
        self.on_refresh()

    def _refresh_tree(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for r in db.get_attendance_today():
            self.tree.insert("", "end", values=(
                r["user_name"],
                r["timestamp"][11:19],
                f'{r["similarity"]:.3f}'
            ))


# ─────────────────────────────────────────────────────────────────────────────
# Users Tab
# ─────────────────────────────────────────────────────────────────────────────

class UsersTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        f = card(self, "👥 Daftar Pengguna Terdaftar", padding=10)
        f.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        f.rowconfigure(0, weight=1)
        f.columnconfigure(0, weight=1)

        cols = ("ID", "Nama", "Tanggal Daftar", "Jumlah Frame")
        self.tree = ttk.Treeview(f, columns=cols, show="headings")
        widths    = [50, 220, 200, 120]
        for c, w in zip(cols, widths):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w)
        sc = ttk.Scrollbar(f, command=self.tree.yview)
        self.tree.config(yscrollcommand=sc.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        sc.grid(row=0, column=1, sticky="ns")

        bf = tk.Frame(f, bg=THEME_PANEL)
        bf.grid(row=1, column=0, columnspan=2, pady=8, sticky="w")
        btn(bf, "🔄  Refresh",       self.refresh,      THEME_ACCENT, 14).pack(side="left", padx=4)
        btn(bf, "🗑️  Hapus Terpilih", self._delete_user, THEME_RED,    16).pack(side="left", padx=4)

        self.refresh()

    def refresh(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for u in db.get_all_users():
            ts = u["registered_at"][:19].replace("T", " ")
            self.tree.insert("", "end", values=(
                u["id"], u["name"], ts, u["frame_count"]
            ))

    def _delete_user(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Info", "Pilih pengguna yang akan dihapus.")
            return
        uid  = self.tree.item(sel[0])["values"][0]
        name = self.tree.item(sel[0])["values"][1]
        if messagebox.askyesno("Konfirmasi", f"Hapus '{name}' beserta riwayat absensinya?"):
            db.delete_user(uid)
            self.refresh()


# ─────────────────────────────────────────────────────────────────────────────
# Report Tab
# ─────────────────────────────────────────────────────────────────────────────

class ReportTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # Stats bar
        stats_f = tk.Frame(self, bg=THEME_BG)
        stats_f.grid(row=0, column=0, sticky="ew", padx=10, pady=(10,4))
        self.stat_cards = {}
        for key, label in [("total_users","Total User"), ("today_count","Hadir Hari Ini"),
                            ("total_records","Total Absensi")]:
            c = tk.Frame(stats_f, bg=THEME_CARD, padx=20, pady=12)
            c.pack(side="left", padx=8)
            tk.Label(c, text=label,  bg=THEME_CARD, fg=THEME_MUTED,
                     font=("Segoe UI", 9)).pack()
            lbl = tk.Label(c, text="–", bg=THEME_CARD, fg=THEME_ACCENT,
                           font=("Segoe UI Bold", 22))
            lbl.pack()
            self.stat_cards[key] = lbl

        # Table
        f = card(self, "📄 Semua Riwayat Absensi", padding=10)
        f.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)
        f.rowconfigure(0, weight=1)
        f.columnconfigure(0, weight=1)

        cols = ("ID", "Nama", "Tanggal", "Waktu", "Similarity", "Foto Bukti")
        self.tree = ttk.Treeview(f, columns=cols, show="headings")
        widths    = [50, 180, 110, 90, 100, 300]
        for c, w in zip(cols, widths):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w)
        sc_y = ttk.Scrollbar(f, command=self.tree.yview)
        sc_x = ttk.Scrollbar(f, orient="horizontal", command=self.tree.xview)
        self.tree.config(yscrollcommand=sc_y.set, xscrollcommand=sc_x.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        sc_y.grid(row=0, column=1, sticky="ns")
        sc_x.grid(row=1, column=0, sticky="ew")

        bf = tk.Frame(f, bg=THEME_PANEL)
        bf.grid(row=2, column=0, columnspan=2, pady=6, sticky="w")
        btn(bf, "🔄  Refresh",       self.refresh,       THEME_ACCENT, 14).pack(side="left", padx=4)
        btn(bf, "💾  Export CSV",    self._export_csv,   THEME_AMBER,  14).pack(side="left", padx=4)

        self.refresh()

    def refresh(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for r in db.get_attendance_all():
            ts  = r["timestamp"]
            date = ts[:10]
            time = ts[11:19]
            self.tree.insert("", "end", values=(
                r["id"], r["user_name"], date, time,
                f'{r["similarity"]:.3f}', r["photo_path"] or ""
            ))
        stats = db.get_attendance_stats()
        for k, lbl in self.stat_cards.items():
            lbl.config(text=str(stats.get(k, "–")))

    def _export_csv(self):
        import csv
        path = os.path.join(DATA_DIR, f"absensi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        rows = db.get_attendance_all()
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ID","Nama","Tanggal","Waktu","Similarity","Foto"])
            for r in rows:
                ts = r["timestamp"]
                writer.writerow([r["id"], r["user_name"], ts[:10], ts[11:19],
                                 f'{r["similarity"]:.3f}', r["photo_path"] or ""])
        messagebox.showinfo("Export Berhasil", f"CSV disimpan:\n{path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main App Window
# ─────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        db.init_db()

        self.title(APP_TITLE)
        self.geometry(f"{APP_WIDTH}x{APP_HEIGHT}")
        self.configure(bg=THEME_BG)
        self.resizable(True, True)
        apply_dark_style(self)

        self._build_header()
        self._build_tabs()

    def _build_header(self):
        hdr = tk.Frame(self, bg=THEME_PANEL, height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="🎯", bg=THEME_PANEL,
                 font=("Segoe UI", 20)).pack(side="left", padx=(18,6), pady=8)
        tk.Label(hdr, text="Sistem Absensi Pengenalan Wajah",
                 bg=THEME_PANEL, fg=THEME_TEXT,
                 font=("Segoe UI Semibold", 14)).pack(side="left")
        tk.Label(hdr, text="Video-Based Face Enrollment",
                 bg=THEME_PANEL, fg=THEME_MUTED,
                 font=("Segoe UI", 9)).pack(side="left", padx=10)

        self.clock_lbl = tk.Label(hdr, bg=THEME_PANEL, fg=THEME_ACCENT,
                                  font=("Segoe UI", 10))
        self.clock_lbl.pack(side="right", padx=18)
        self._tick()

    def _tick(self):
        self.clock_lbl.config(text=datetime.now().strftime("%A, %d %B %Y  %H:%M:%S"))
        self.after(1000, self._tick)

    def _build_tabs(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=0, pady=0)

        self.report_tab = ReportTab(nb)
        self.users_tab  = UsersTab(nb)

        self.enroll_tab = EnrollmentTab(nb)
        self.attend_tab = AttendanceTab(nb, on_refresh=self._on_attendance_refresh)

        nb.add(self.enroll_tab,  text="  📝  Enrollment  ")
        nb.add(self.attend_tab,  text="  ✅  Absensi  ")
        nb.add(self.users_tab,   text="  👥  Pengguna  ")
        nb.add(self.report_tab,  text="  📊  Laporan  ")

    def _on_attendance_refresh(self):
        self.report_tab.refresh()
        self.users_tab.refresh()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
