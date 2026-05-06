$TaskName = "NQ_Data_Daily_Update"
$ActionExecutable = "C:\Users\chadm\AppData\Local\Microsoft\WindowsApps\python.exe"
$ActionScript = "c:\Users\chadm\OneDrive\Desktop\Quant Lab\NQ Futures Historical Data\update_nq_data.py"
$TriggerTime = "08:00:00"

Write-Host "Registering scheduled task for NQ Data Update..."

# Unregister if already exists to ensure fresh start
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

# Create Action with Working Directory set to the script folder
$Action = New-ScheduledTaskAction -Execute $ActionExecutable -Argument $ActionScript -WorkingDirectory (Split-Path $ActionScript -Parent)

# Create Trigger (Daily at 8am)
$Trigger = New-ScheduledTaskTrigger -Daily -At $TriggerTime

# Create Principal (Run as current user, requires user to be logged in if interactive)
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive

# Register Task
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal

Write-Host "Success: Task '$TaskName' has been scheduled to run daily at $TriggerTime."
Write-Host "Script path: $ActionScript"
