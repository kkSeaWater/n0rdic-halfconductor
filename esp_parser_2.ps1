param(
  [string]$Port = "COM4",
  [int]$Baud = 115200,
  [int]$IntervalMs = 1000,
  [int]$DurationSec = 300,
  [int]$WarmupSec = 2,        # wait after thread start before first tick
  [switch]$ForceChild,        # --ForceChild to disable router eligibility
  [switch]$ResetCounters      # --ResetCounters to clear MAC counters at start
)

# --- Default values if none provided ---
if (-not $Port)         { $Port = "COM4" }
if (-not $Baud)         { $Baud = 115200 }
if (-not $IntervalMs)   { $IntervalMs = 1000 }
if (-not $DurationSec)  { $DurationSec = 300 }
if (-not $WarmupSec)    { $WarmupSec = 2 }
if (-not $ForceChild)   { $ForceChild = $true }
if (-not $ResetCounters){ $ResetCounters = $true }


# ---------- Setup output ----------
$stamp  = Get-Date -Format "yyyyMMdd_HHmmss"
$Base   = (Get-Location).Path
$LogTxt = Join-Path $Base "esp_child_log_$stamp.txt"
$LogCsv = Join-Path $Base "esp_child_telemetry_$stamp.csv"
$AnsiRx = '\x1b\[[0-9;]*m'

# ---------- CSV header ----------
$CsvHeader = @(
  'timestamp','state','event','parent_rloc16','parent_ipv6','lqi_in','lqi_out','age_s',
  'tx_total','rx_total','tx_err_cca','tx_retry','rx_err_fcs','rtt_ms','note'
) -join ','

# ---------- Serial helpers ----------
function Open-Serial {
  param([string]$Port,[int]$Baud)
  if ($script:sp) { try { if ($sp.IsOpen){$sp.Close()}; $sp.Dispose() } catch {} }
  $script:sp = New-Object System.IO.Ports.SerialPort $Port,$Baud,"None",8,"One"
  $sp.NewLine="`r`n"; $sp.Handshake="None"; $sp.DtrEnable=$true; $sp.RtsEnable=$false; $sp.ReadTimeout=800
  $sp.Open()
}

function Write-Line($s){ $sp.WriteLine($s) }

function Read-UntilDoneOrError([int]$timeoutMs=3000){
  $deadline = [datetime]::UtcNow.AddMilliseconds($timeoutMs)
  $lines = New-Object System.Collections.Generic.List[string]
  while([datetime]::UtcNow -lt $deadline){
    try {
      $line = $sp.ReadLine()
      if ($line -ne $null){
        $clean = ($line -replace $AnsiRx,'').Trim()
        if ($clean.Length){ $lines.Add($clean) }
        if ($clean -eq 'Done' -or $clean -like 'Error*'){ break }
      }
    } catch { Start-Sleep -Milliseconds 20 }
  }
  return ,$lines
}

function OT([string]$cmd,[int]$timeoutMs=3000){
  Write-Line $cmd
  (Read-UntilDoneOrError -timeoutMs $timeoutMs)
}

# ---------- Small parsers ----------
function Get-State {
  (OT 'state') |
    Where-Object { $_ -match '^(disabled|detached|child|router|leader)$' } |
    Select-Object -First 1
}

function Get-ParentRloc16Hex {
  $p = (OT 'parent')
  $rl = ($p | Where-Object { $_ -match '(Rloc|Rloc16):\s*([0-9a-fA-F]{1,4})' } | ForEach-Object { $Matches[2] }) | Select-Object -First 1
  if (-not $rl){ return $null }
  # Zero-pad to 4 hex, lowercase
  return ('{0:x4}' -f [int]("0x$rl")).ToLower()
}

function Get-MyRlocAddr {
  # pick the address ending with :0:ff:fe00:####
  $ips = (OT 'ipaddr')
  ($ips | Where-Object { $_ -match '^[0-9a-f:]+:0:ff:fe00:[0-9a-f]{1,4}$' } | Select-Object -First 1)
}

function Build-ParentIPv6([string]$parentRloc16){
  $myRloc = Get-MyRlocAddr
  if (-not $myRloc){ return $null }
  # Extract prefix "xxxx:...:0:ff:fe00:"
  $prefix = ($myRloc -replace '([0-9a-f:]+:0:ff:fe00:)[0-9a-f]{1,4}$','$1')
  if (-not $prefix){ return $null }
  return ($prefix + $parentRloc16).ToLower()
}

function Get-Counters {
  $c = (OT 'counters mac')
  @{
    tx_total   = ($c | ?{$_ -match '^TxTotal:\s+(\d+)'} | %{$matches[1]} | Select -First 1)
    rx_total   = ($c | ?{$_ -match '^RxTotal:\s+(\d+)'} | %{$matches[1]} | Select -First 1)
    tx_err_cca = ($c | ?{$_ -match '^TxErrCca:\s+(\d+)'} | %{$matches[1]} | Select -First 1)
    tx_retry   = ($c | ?{$_ -match '^TxRetry:\s+(\d+)'} | %{$matches[1]} | Select -First 1)
    rx_err_fcs = ($c | ?{$_ -match '^RxErrFcs:\s+(\d+)'} | %{$matches[1]} | Select -First 1)
  }
}

function Get-ParentInfo {
  $p = (OT 'parent')
  @{
    lqi_in  = ($p | ?{$_ -match 'Link\s*Quality\s*In:\s*(\d+)'}  | %{$matches[1]} | Select -First 1)
    lqi_out = ($p | ?{$_ -match 'Link\s*Quality\s*Out:\s*(\d+)'} | %{$matches[1]} | Select -First 1)
    age_s   = ($p | ?{$_ -match '^Age:\s*(\d+)'}                 | %{$matches[1]} | Select -First 1)
  }
}

function Ping([string]$addr,[int]$timeoutMs=1500){
  if (-not $addr){ return $null }
  $out = OT "ping $addr" $timeoutMs
  $rtt = ($out | Where-Object { $_ -match 'time[=\s](\d+)\s*ms' } | ForEach-Object { $matches[1] } | Select-Object -First 1)
  if ($rtt){ return [int]$rtt } else { return $null }
}

# ---------- Start logging ----------
"# ESP Child Telemetry started $stamp" | Out-File $LogTxt -Encoding UTF8
$CsvHeader | Out-File $LogCsv -Encoding UTF8

# Track previous values to detect events
$prevState = $null
$prevParentRloc16 = $null

try {
  Open-Serial -Port $Port -Baud $Baud

  # Clean start
  OT 'thread stop' | Out-Null
  OT 'ifconfig up' | Out-Null
  if ($ForceChild){ OT 'routereligible disable' | Out-Null }
  if ($ResetCounters){ OT 'counters mac reset' | Out-Null }
  OT 'thread start' | Out-Null

  if ($WarmupSec -gt 0) { Start-Sleep -Seconds $WarmupSec }

  $ticks = [math]::Ceiling($DurationSec*1000.0 / $IntervalMs)

  for ($i=0; $i -lt $ticks; $i++){
    $ts = Get-Date

    $state = Get-State
    if (-not $state) { $state = '' }

    $parentRloc16 = Get-ParentRloc16Hex
    $parentIPv6 = if ($parentRloc16){ Build-ParentIPv6 $parentRloc16 } else { $null }

    $pinfo = Get-ParentInfo
    $c     = Get-Counters
    $rtt   = Ping $parentIPv6
    $rttStr = if ($null -ne $rtt) { [string]$rtt } else { '' }

    # --- Event detection ---
    $events = @()
    if (-not $state) { $events += 'state_blank' }

    if ($prevState -ne 'detached' -and $state -eq 'detached') { $events += 'detached_start' }
    if ($prevState -eq 'detached' -and $state -eq 'child')    { $events += 'reattached' }

    if ($prevParentRloc16 -and $parentRloc16 -and ($prevParentRloc16 -ne $parentRloc16) -and $state -eq 'child') {
      $events += 'parent_switch'
    }

    if (-not $parentIPv6) { $events += 'parent_unreachable' }

    $event = ($events -join '|')

    $note  = if (-not $parentIPv6) { "no_parent_ipv6" } else { "" }

    # TXT line
    $lineTxt = "[{0}] state={1} event={2} parent_rloc16={3} parent_ipv6={4} lqi_in={5} lqi_out={6} age={7}s tx={8} rx={9} cca={10} retry={11} fcs={12} rtt_ms={13} {14}" -f `
      ($ts.ToString("HH:mm:ss.fff")),$state,$event,$parentRloc16,$parentIPv6,$pinfo.lqi_in,$pinfo.lqi_out,$pinfo.age_s,`
      $c.tx_total,$c.rx_total,$c.tx_err_cca,$c.tx_retry,$c.rx_err_fcs,$rttStr,$note
    $lineTxt | Out-File $LogTxt -Append -Encoding UTF8

    # CSV line
    $csv = @(
      $ts.ToString("o"), $state, $event, $parentRloc16, $parentIPv6, $pinfo.lqi_in, $pinfo.lqi_out, $pinfo.age_s,
      $c.tx_total, $c.rx_total, $c.tx_err_cca, $c.tx_retry, $c.rx_err_fcs, $rttStr, $note
    ) -join ','
    $csv | Out-File $LogCsv -Append -Encoding UTF8

    # update prevs
    $prevState = $state
    $prevParentRloc16 = $parentRloc16

    Start-Sleep -Milliseconds $IntervalMs
  }
}
finally {
  try { if ($sp){ if ($sp.IsOpen){ $sp.Close() }; $sp.Dispose() } } catch {}
  $footer = "Logs saved:`n  TXT: $LogTxt`n  CSV: $LogCsv"
  $footer | Out-File $LogTxt -Append -Encoding UTF8
  Write-Host $footer
}