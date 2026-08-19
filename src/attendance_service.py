import csv
import os
from datetime import datetime


class AttendanceService:

    def __init__(
        self,
        attendance_file="attendance/attendance.csv"
    ):

        self.attendance_file = attendance_file

        os.makedirs(
            os.path.dirname(
                self.attendance_file
            ),
            exist_ok=True
        )

        self._create_file_if_needed()


    # ======================================
    # Create CSV
    # ======================================

    def _create_file_if_needed(self):

        if not os.path.exists(
            self.attendance_file
        ):

            with open(
                self.attendance_file,
                "w",
                newline=""
            ) as file:

                writer = csv.writer(file)

                writer.writerow(
                    [
                        "Student_ID",
                        "Name",
                        "Date",
                        "Time",
                        "Status"
                    ]
                )


    # ======================================
    # Check today's attendance
    # ======================================

    def already_marked_today(
        self,
        student_id
    ):

        today = datetime.now().strftime(
            "%Y-%m-%d"
        )

        with open(
            self.attendance_file,
            "r",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                if (
                    row["Student_ID"] == student_id
                    and row["Date"] == today
                ):

                    return True

        return False


    # ======================================
    # Mark attendance
    # ======================================

    def mark_attendance(
        self,
        student_id,
        name
    ):

        # Prevent duplicate attendance

        if self.already_marked_today(
            student_id
        ):

            return {
                "success": False,
                "already_marked": True,
                "message":
                    f"{name} is already "
                    "marked present today."
            }


        now = datetime.now()

        date = now.strftime(
            "%Y-%m-%d"
        )

        time = now.strftime(
            "%H:%M:%S"
        )


        # Save record

        with open(
            self.attendance_file,
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    student_id,
                    name,
                    date,
                    time,
                    "Present"
                ]
            )


        return {
            "success": True,
            "already_marked": False,
            "message":
                f"Attendance marked for {name}.",
            "student_id": student_id,
            "name": name,
            "date": date,
            "time": time,
            "status": "Present"
        }