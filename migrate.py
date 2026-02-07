#!/usr/bin/env python
"""
Database Migration Script
Creates all required tables for the Hiring and Assessment Portal
"""

import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE = os.getenv("DATABASE_PATH", os.path.join(os.getcwd(), "users.db"))


def migrate():
    """Create all database tables"""
    print(f"Connecting to database: {DATABASE}")
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()

    # ── Users table ────────────────────────────────────────────────
    print("Creating users table...")
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  role TEXT NOT NULL DEFAULT 'candidate',
                  face_embedding TEXT,
                  resume_filename TEXT,
                  resume_original_name TEXT,
                  resume_uploaded_at TIMESTAMP,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # Add columns if they don't exist (for existing databases)
    columns_to_add = [
        ("email", "TEXT"),
        ("role", "TEXT NOT NULL DEFAULT 'candidate'"),
        ("resume_filename", "TEXT"),
        ("resume_original_name", "TEXT"),
        ("resume_uploaded_at", "TIMESTAMP"),
    ]
    for col_name, col_type in columns_to_add:
        try:
            c.execute(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}")
            print(f"  Added column: {col_name}")
        except sqlite3.OperationalError:
            pass  # column already exists

    # ── OTP codes table ────────────────────────────────────────────
    print("Creating otp_codes table...")
    c.execute('''CREATE TABLE IF NOT EXISTS otp_codes
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  email TEXT NOT NULL,
                  otp TEXT NOT NULL,
                  purpose TEXT NOT NULL DEFAULT 'register',
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  expires_at TIMESTAMP NOT NULL,
                  used INTEGER DEFAULT 0)''')

    # ── Companies table ────────────────────────────────────────────
    print("Creating companies table...")
    c.execute('''CREATE TABLE IF NOT EXISTS companies
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT UNIQUE NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # ── Jobs table ─────────────────────────────────────────────────
    print("Creating jobs table...")
    c.execute('''CREATE TABLE IF NOT EXISTS jobs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  company_id INTEGER,
                  created_by_user_id INTEGER,
                  title TEXT NOT NULL,
                  description TEXT,
                  skills_json TEXT,
                  modules_json TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(company_id) REFERENCES companies(id),
                  FOREIGN KEY(created_by_user_id) REFERENCES users(id))''')

    # ── Assessments table ──────────────────────────────────────────
    print("Creating assessments table...")
    c.execute('''CREATE TABLE IF NOT EXISTS assessments
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  job_id INTEGER,
                  candidate_user_id INTEGER,
                  invited_email TEXT,
                  status TEXT NOT NULL DEFAULT 'Pending',
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  updated_at TIMESTAMP,
                  FOREIGN KEY(job_id) REFERENCES jobs(id),
                  FOREIGN KEY(candidate_user_id) REFERENCES users(id))''')

    # ── Proctor logs table ─────────────────────────────────────────
    print("Creating proctor_logs table...")
    c.execute('''CREATE TABLE IF NOT EXISTS proctor_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  assessment_id INTEGER,
                  type TEXT NOT NULL,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  payload_json TEXT,
                  FOREIGN KEY(user_id) REFERENCES users(id),
                  FOREIGN KEY(assessment_id) REFERENCES assessments(id))''')

    # ── Candidate reports table ────────────────────────────────────
    print("Creating candidate_reports table...")
    c.execute('''CREATE TABLE IF NOT EXISTS candidate_reports
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  candidate_user_id INTEGER,
                  assessment_id INTEGER,
                  report_json TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(candidate_user_id) REFERENCES users(id),
                  FOREIGN KEY(assessment_id) REFERENCES assessments(id))''')

    # ── Applications table ─────────────────────────────────────────
    print("Creating applications table...")
    c.execute('''CREATE TABLE IF NOT EXISTS applications
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  candidate_user_id INTEGER NOT NULL,
                  job_id INTEGER NOT NULL,
                  status TEXT NOT NULL DEFAULT 'Applied',
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  UNIQUE(candidate_user_id, job_id),
                  FOREIGN KEY(candidate_user_id) REFERENCES users(id),
                  FOREIGN KEY(job_id) REFERENCES jobs(id))''')

    conn.commit()
    
    # ── Verify tables ──────────────────────────────────────────────
    print("\n--- Verifying tables ---")
    c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = c.fetchall()
    print("Tables created:")
    for table in tables:
        print(f"  ✓ {table[0]}")
    
    conn.close()
    print(f"\n✅ Migration complete! Database: {DATABASE}")


if __name__ == "__main__":
    migrate()
