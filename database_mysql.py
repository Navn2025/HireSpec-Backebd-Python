import mysql.connector
from mysql.connector import pooling
import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '3306')),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'interview_platform_db'),
    'autocommit': True
}

# Connection pool
connection_pool = None

def init_database():
    """Initialize MySQL database connection pool"""
    global connection_pool
    try:
        connection_pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name="interview_pool",
            pool_size=5,
            pool_reset_session=True,
            **DB_CONFIG
        )
        print("✅ MySQL database connected successfully (Python)")
        return True
    except mysql.connector.Error as err:
        print(f"❌ MySQL connection error: {err}")
        print("   Falling back to SQLite")
        return False

def get_connection():
    """Get a database connection from the pool"""
    if connection_pool:
        return connection_pool.get_connection()
    return None

def execute_query(query, params=None, fetch=True):
    """Execute a database query"""
    try:
        conn = get_connection()
        if not conn:
            raise Exception("Database not available")
        
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, params or ())
        
        if fetch:
            result = cursor.fetchall()
        else:
            result = cursor.lastrowid
            conn.commit()
        
        cursor.close()
        conn.close()
        return result
    except Exception as e:
        print(f"Database query error: {e}")
        raise e

# Helper functions
class DB:
    @staticmethod
    def find_user_by_email(email):
        """Find user by email"""
        result = execute_query(
            "SELECT * FROM users WHERE email = %s LIMIT 1",
            (email,)
        )
        return result[0] if result else None
    
    @staticmethod
    def find_user_by_username(username):
        """Find user by username"""
        result = execute_query(
            "SELECT * FROM users WHERE username = %s LIMIT 1",
            (username,)
        )
        return result[0] if result else None
    
    @staticmethod
    def create_user(username, email, password, role='candidate'):
        """Create a new user"""
        return execute_query(
            "INSERT INTO users (username, email, password, role) VALUES (%s, %s, %s, %s)",
            (username, email, password, role),
            fetch=False
        )
    
    @staticmethod
    def update_user_face_embedding(user_id, embedding):
        """Update user's face embedding"""
        execute_query(
            "UPDATE users SET face_embedding = %s WHERE id = %s",
            (embedding, user_id),
            fetch=False
        )
    
    @staticmethod
    def save_otp(email, otp, purpose='register', expires_minutes=10):
        """Save OTP code"""
        from datetime import datetime, timedelta
        expires_at = datetime.now() + timedelta(minutes=expires_minutes)
        return execute_query(
            "INSERT INTO otp_codes (email, otp, purpose, expires_at) VALUES (%s, %s, %s, %s)",
            (email, otp, purpose, expires_at),
            fetch=False
        )
    
    @staticmethod
    def verify_otp(email, otp, purpose='register'):
        """Verify OTP code"""
        from datetime import datetime
        result = execute_query(
            """SELECT * FROM otp_codes 
               WHERE email = %s AND otp = %s AND purpose = %s 
               AND used = FALSE AND expires_at > %s 
               ORDER BY created_at DESC LIMIT 1""",
            (email, otp, purpose, datetime.now())
        )
        
        if result:
            # Mark OTP as used
            execute_query(
                "UPDATE otp_codes SET used = TRUE WHERE id = %s",
                (result[0]['id'],),
                fetch=False
            )
            return True
        return False
    
    @staticmethod
    def create_company(name, description=None):
        """Create a new company"""
        return execute_query(
            "INSERT INTO companies (name, description) VALUES (%s, %s)",
            (name, description),
            fetch=False
        )
    
    @staticmethod
    def create_job(company_id, created_by, title, description):
        """Create a new job posting"""
        return execute_query(
            """INSERT INTO jobs (company_id, created_by_user_id, title, description) 
               VALUES (%s, %s, %s, %s)""",
            (company_id, created_by, title, description),
            fetch=False
        )
    
    @staticmethod
    def create_application(candidate_id, job_id):
        """Create a job application"""
        return execute_query(
            "INSERT INTO applications (candidate_user_id, job_id) VALUES (%s, %s)",
            (candidate_id, job_id),
            fetch=False
        )
    
    @staticmethod
    def get_user_applications(user_id):
        """Get all applications for a user"""
        return execute_query(
            """SELECT app.*, j.title as job_title, c.name as company_name
               FROM applications app
               JOIN jobs j ON app.job_id = j.id
               LEFT JOIN companies c ON j.company_id = c.id
               WHERE app.candidate_user_id = %s
               ORDER BY app.applied_at DESC""",
            (user_id,)
        )
    
    @staticmethod
    def log_activity(user_id, action, entity_type=None, entity_id=None, description=None):
        """Log user activity"""
        return execute_query(
            """INSERT INTO activity_logs (user_id, action, entity_type, entity_id, description) 
               VALUES (%s, %s, %s, %s, %s)""",
            (user_id, action, entity_type, entity_id, description),
            fetch=False
        )
