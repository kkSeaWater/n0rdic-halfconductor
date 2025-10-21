<#
ChildTelemetry_Risk.ps1
PowerShell 5.1-compatible
- Logs OT telemetry to Desktop\risk_logs_*.txt and risk_telemetry_*.csv
- Predicts link degradation (LQI / RTT rolling stats)
- Triggers proactive reattach: thread stop → ifconfig up → thread start
#>

# ===================== CONFIG =====================
$Port             = "COM8"
$Baud             = 115200
$IntervalMs       = 1000
$DurationSec      = 300
$WarmupSec        = 2
$UseOtPrefix      = $true
$ResetCounters    = $true
$ForceChild       = $true

# Predictive thresholds - adjusted for OpenThread LQI range 0-3 (3=best)
$LqiFloor         = 2      # Risk if mean < 2
$LqiJitterMin     = 0.5    # Risk if std dev > 0.5
$RttCeilMs        = 70
$DwellSec         = 3
$CooldownSec      = 60
$AttachTimeoutSec = 10
# ==================================================

$stamp  = Get-Date -Format "yyyyMMdd_HHmmss"
$Base   = Join-Path $env:USERPROFILE "Desktop"
$LogTxt = Join-Path $Base "risk_logs_$stamp.txt"
$LogCsv = Join-Path $Base "risk_telemetry_$stamp.csv"

"[$(Get-Date -Format 'HH:mm:ss.fff')] Starting ChildTelemetry_Risk.ps1" |
  Out-File $LogTxt -Encoding UTF8

# ---------- helpers ----------
function Nz([object]$v,[object]$alt){ if($null -ne $v -and $v -ne ""){ $v }else{ $alt } }

$script:sp = $null
function Open-Serial {
  if($script:sp){ try{ if($sp.IsOpen){$sp.Close()} $sp.Dispose() }catch{} }
  $script:sp = New-Object System.IO.Ports.SerialPort $Port,$Baud,"None",8,"One"
  $sp.NewLine="`r`n";$sp.Handshake="None";$sp.DtrEnable=$true;$sp.RtsEnable=$false
  $sp.ReadTimeout=800;$sp.WriteTimeout=800;$sp.Open()
}

$ansi='\x1b\[[0-9;]*m'
function Read-UntilDoneOrError([int]$timeoutMs=2000){
  $deadline = [datetime]::UtcNow.AddMilliseconds($timeoutMs)
  $lines = New-Object System.Collections.Generic.List[string]
  while([datetime]::UtcNow -lt $deadline){
    try {
      $line = $sp.ReadLine()
      if ($line -ne $null){
        $clean = ($line -replace $ansi,'').Trim()
        if ($clean.Length){ $lines.Add($clean) }
        if ($clean -eq 'Done' -or $clean -like 'Error*'){ break }
      }
    } catch { Start-Sleep -Milliseconds 20 }
  }
  return $lines
}

function OT([string]$cmd,[int]$timeoutMs=2000){
  if(-not $script:sp -or -not $sp.IsOpen){throw "Serial not open"}
  $prefix=$(if($UseOtPrefix){"ot "}else{""});$full=$prefix+$cmd
  $sp.DiscardInBuffer();$sp.WriteLine($full)
  $lines = Read-UntilDoneOrError -timeoutMs $timeoutMs
  $lines|?{$_-ne""}
}

function Get-State{
  $s=OT "state"
  foreach($l in $s){if($l -match '^(detached|child|router|leader|disabled)$'){return $matches[1]}}
  return ($s|Select-Object -Last 1)
}
function Get-ParentInfo{
  $p=OT "parent"
  $ext="";$rloc="";$lqi_in=$null;$lqi_out=$null;$age=$null
  foreach($l in $p){
    if($l -match '(Ext|Extended) Addr:\s*([0-9a-fA-F]{16})'){$ext=$matches[2].ToLower()}
    if($l -match '(Rloc|Rloc16):\s*([0-9a-fA-F]{1,4})'){$rloc=('{0:x4}' -f [int]("0x"+$matches[2]))}
    if($l -match 'Link\s*Quality\s*In:\s*(\d+)'){$lqi_in=[int]$matches[1]}
    if($l -match 'Link\s*Quality\s*Out:\s*(\d+)'){$lqi_out=[int]$matches[1]}
    if($l -match '^Age:\s*(\d+)'){$age=[int]$matches[1]}
  }
  [pscustomobject]@{ExtAddr=$ext;RLOC16=$rloc;LQI_in=$lqi_in;LQI_out=$lqi_out;Age=$age}
}
function Get-MyRlocAddr{
  $ips=OT "ipaddr"
  $ips|?{$_ -match '^[0-9a-f:]+:0:ff:fe00:[0-9a-f]{1,4}$'}|Select-Object -First 1
}
function Build-ParentIPv6([string]$rloc){
  $my=Get-MyRlocAddr;if(-not $my){return $null}
  $prefix=($my -replace '([0-9a-f:]+:0:ff:fe00:)[0-9a-f]{1,4}$','$1')
  if(-not $prefix){return $null};($prefix+$rloc).ToLower()
}
function Get-Counters{
  $c=OT "counters mac"
  @{
    tx_total=($c|?{$_ -match '^TxTotal:\s+(\d+)'}|%{$matches[1]}|Select -First 1)
    rx_total=($c|?{$_ -match '^RxTotal:\s+(\d+)'}|%{$matches[1]}|Select -First 1)
    tx_err_cca=($c|?{$_ -match '^TxErrCca:\s+(\d+)'}|%{$matches[1]}|Select -First 1)
    tx_retry=($c|?{$_ -match '^TxRetry:\s+(\d+)'}|%{$matches[1]}|Select -First 1)
    rx_err_fcs=($c|?{$_ -match '^RxErrFcs:\s+(\d+)'}|%{$matches[1]}|Select -First 1)
  }
}
function Get-PingRTT([string]$addr){
  if(-not $addr){return $null}
  $out=OT "ping $addr" 3000
  $all=($out -join ' ')
  if($all -match 'time[=/]\s?([0-9\.]+)\s?ms'){[int]([double]$matches[1])}else{$null}
}
function Reset-MacCounters{OT "counters mac reset"|Out-Null}
function Force-ChildMode{OT "mode rsd"|Out-Null}

function Reattach-StopStart{
  param([string]$Reason="predictive_switch")
  $t0=Get-Date;$oldP=Get-ParentInfo;$oldS=Get-State
  OT "thread stop"|Out-Null;OT "ifconfig up"|Out-Null;OT "thread start"|Out-Null
  Start-Sleep -Milliseconds 500
  $deadline=(Get-Date).AddSeconds($AttachTimeoutSec)
  $state="";$newP=$null
  do{
    $state=Get-State
    if($state -in @("child","router")){
      $newP=Get-ParentInfo
      if($newP.ExtAddr -and ($newP.ExtAddr -ne $oldP.ExtAddr -or $oldP.ExtAddr -eq "")){break}
    }
    Start-Sleep -Milliseconds 250
  }while((Get-Date) -lt $deadline)
  $elapsed=[math]::Round(((Get-Date)-$t0).TotalSeconds,2)
  $ok=($state -in @("child","router"))
  $line="[$($t0.ToString('HH:mm:ss.fff'))] event=$Reason ok=$ok elapsed=${elapsed}s old_parent=$($oldP.ExtAddr)/$($oldP.RLOC16)/LQI$($oldP.LQI_in) new_parent=$($newP.ExtAddr)/$($newP.RLOC16)/LQI$($newP.LQI_in) stateAfter=$state"
  $line|Out-File $LogTxt -Append -Encoding UTF8
  [pscustomobject]@{ok=$ok;elapsed=$elapsed;newParent=$newP;state=$state}
}

# ---------- start ----------
Open-Serial
if($ForceChild){Force-ChildMode}
if($ResetCounters){Reset-MacCounters}
try{if(Get-State -eq "disabled"){OT "ifconfig up"|Out-Null;OT "thread start"|Out-Null}}catch{}
Start-Sleep -Seconds $WarmupSec

# CSV header
$CsvHeader='timestamp,state,event,parent_rloc16,parent_ipv6,lqi_in,lqi_out,age_s,tx_total,rx_total,tx_err_cca,tx_retry,rx_err_fcs,rtt_ms,note'
$CsvHeader|Out-File $LogCsv -Encoding UTF8

$lqiHist=New-Object System.Collections.Generic.Queue[double]
$rttHist=New-Object System.Collections.Generic.Queue[double]
$winSec=10;$winCount=[Math]::Ceiling($winSec*(1000.0/$IntervalMs))
$lastSwitchAt=(Get-Date).AddSeconds(-9999);$dwellStart=$null
$prevState="";$prevParent=""

$tsStart=Get-Date;$endAt=$tsStart.AddSeconds($DurationSec)
try{
  while([datetime]::UtcNow -lt $endAt.ToUniversalTime()){
    $ts=Get-Date
    $state=Get-State;$pinfo=Get-ParentInfo
    $prloc=$pinfo.RLOC16;$pIPv6=if($prloc){Build-ParentIPv6 $prloc}else{$null}
    $c=Get-Counters
    $rtt=$null;if($pIPv6){$rtt=Get-PingRTT -addr $pIPv6}

    if($pinfo.LQI_in -ne $null -and $pinfo.LQI_in -ne ""){
      $lqiHist.Enqueue([double]$pinfo.LQI_in);while($lqiHist.Count -gt $winCount){$null=$lqiHist.Dequeue()}
    }
    if($rtt -ne $null){$rttHist.Enqueue([double]$rtt);while($rttHist.Count -gt $winCount){$null=$rttHist.Dequeue()}}

    $lqiArr=$lqiHist.ToArray();$rttArr=$rttHist.ToArray()
    $lqiMean=$null;$lqiStd=$null;$rttMean=$null
    if($lqiArr.Length -gt 0){
      $lqiMean=[Math]::Round((($lqiArr|Measure-Object -Average).Average),2)
      if($lqiArr.Length -ge 2){
        $m=($lqiArr|Measure-Object -Average).Average;$sum=0.0
        foreach($v in $lqiArr){$sum+=($v-$m)*($v-$m)};$lqiStd=[Math]::Round([Math]::Sqrt($sum/$lqiArr.Length),2)
      }
    }
    if($rttArr.Length -gt 0){$rttMean=[Math]::Round((($rttArr|Measure-Object -Average).Average),2)}

    $events=@()
    if($prevState -ne 'detached' -and $state -eq 'detached'){$events+='detached_start'}
    if($prevState -eq 'detached' -and ($state -eq 'child' -or $state -eq 'router')){$events+='reattached'}
    if($prevParent -and $prloc -and ($prevParent -ne $prloc) -and $state -ne 'detached'){$events+='parent_switch'}
    if(-not $pIPv6){$events+='no_parent_ipv6'}

    $risk=$false
    if($lqiMean -ne $null -and $lqiMean -lt $LqiFloor){$risk=$true}
    if($lqiStd  -ne $null -and $lqiStd  -gt $LqiJitterMin){$risk=$true}
    if($rttMean -ne $null -and $rttMean -gt $RttCeilMs){$risk=$true}

    $now=Get-Date;$since=($now-$lastSwitchAt).TotalSeconds
    $eligible=($since -ge $CooldownSec) -and ($state -ne "detached")
    if($risk -and $eligible){
      if(-not $dwellStart){$dwellStart=$now}
      if((($now-$dwellStart).TotalSeconds) -ge $DwellSec){
        $events+='predictive_switch'
        $res=Reattach-StopStart -Reason "predictive_switch"
        if($res.ok){$lastSwitchAt=Get-Date}
        $dwellStart=$null
        $state=Get-State;$pinfo=Get-ParentInfo;$prloc=$pinfo.RLOC16
        $pIPv6=if($prloc){Build-ParentIPv6 $prloc}else{$null}
      }
    }else{$dwellStart=$null}

    $note=@();if($lqiMean -ne $null){$note+="lqiMean=$lqiMean"}
    if($lqiStd -ne $null){$note+="lqiStd=$lqiStd"}
    if($rttMean -ne $null){$note+="rttMean=$rttMean"}
    if(-not $eligible){$note+="cooldown="+[int]$since}
    $noteStr=($note -join ';')

    $eventStr=($events -join '|')
    $lineTxt="[$($ts.ToString('HH:mm:ss.fff'))] state=$state event=$eventStr parent_rloc16=$prloc parent_ipv6=$pIPv6 LQI_in=$($pinfo.LQI_in) LQI_out=$($pinfo.LQI_out) age_s=$($pinfo.Age) tx=$($c.tx_total) rx=$($c.rx_total) cca=$($c.tx_err_cca) retry=$($c.tx_retry) fcs=$($c.rx_err_fcs) rtt_ms=$rtt $noteStr"
    $lineTxt|Out-File $LogTxt -Append -Encoding UTF8

    $csv=("$($ts.ToString('o')),$state,$eventStr,$prloc,$pIPv6,$($pinfo.LQI_in),$($pinfo.LQI_out),$($pinfo.Age),$($c.tx_total),$($c.rx_total),$($c.tx_err_cca),$($c.tx_retry),$($c.rx_err_fcs),$rtt,$noteStr")
    $csv|Out-File $LogCsv -Append -Encoding UTF8

    $prevState=$state;$prevParent=$prloc
    Start-Sleep -Milliseconds $IntervalMs
  }
}
finally{
  try{if($script:sp){if($sp.IsOpen){$sp.Close()}$sp.Dispose()}}catch{}
  "[$(Get-Date -Format 'HH:mm:ss.fff')] Done.`nTXT: $LogTxt`nCSV: $LogCsv"|Out-File $LogTxt -Append -Encoding UTF8
  Write-Host "Logs saved:`n  TXT: $LogTxt`n  CSV: $LogCsv"
}