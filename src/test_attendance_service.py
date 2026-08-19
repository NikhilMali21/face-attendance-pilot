from attendance_service import AttendanceService


print("==========================================")
print(" TESTING ATTENDANCE SERVICE")
print("==========================================")


attendance = AttendanceService()


# ==========================================
# Test 1
# ==========================================

result = attendance.mark_attendance(
    "TEST001",
    "Test Student"
)

print()
print("Test 1:")
print(result)


# ==========================================
# Test 2
# ==========================================

result = attendance.mark_attendance(
    "TEST001",
    "Test Student"
)

print()
print("Test 2:")
print(result)


print()
print("Attendance service test completed.")