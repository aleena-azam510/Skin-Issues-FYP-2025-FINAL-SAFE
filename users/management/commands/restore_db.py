import os
from django.core.management.base import BaseCommand
from django.db import connection

class Command(BaseCommand):
    help = "Restore database backup to Supabase"

    def handle(self, *args, **kwargs):
        # Path to your SQL backup file
        file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "fyp-database-backup-final.sql")
        
        if not os.path.exists(file_path):
            self.stdout.write(self.style.ERROR(f"Backup file not found at {file_path}"))
            return

        self.stdout.write(f"Restoring database from {file_path} ...")

        # Read file in chunks to avoid memory issues
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            sql = f.read()

        with connection.cursor() as cursor:
            cursor.execute(sql)

        self.stdout.write(self.style.SUCCESS("Database restored successfully!"))
