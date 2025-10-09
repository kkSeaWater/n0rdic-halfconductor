param(
  [Parameter(Mandatory)]
  [string]$Hex
)

$T = @{
  CHANNEL=0; PANID=1; EXTPANID=2; NETWORKNAME=3; PSKC=4; NETWORKKEY=5; NETWORK_KEY_SEQUENCE=6
  MESHLOCALPREFIX=7; STEERING_DATA=8; BORDER_AGENT_RLOC=9; COMMISSIONER_ID=10; COMM_SESSION_ID=11
  SECURITYPOLICY=12; GET=13; ACTIVETIMESTAMP=14; COMMISSIONER_UDP_PORT=15; STATE=16; JOINER_DTLS=17
  JOINER_UDP_PORT=18; JOINER_IID=19; JOINER_RLOC=20; JOINER_ROUTER_KEK=21; PROVISIONING_URL=32
  VENDOR_NAME_TLV=33; VENDOR_MODEL_TLV=34; VENDOR_SW_VERSION_TLV=35; VENDOR_DATA_TLV=36
  VENDOR_STACK_VERSION_TLV=37; UDP_ENCAPSULATION_TLV=48; IPV6_ADDRESS_TLV=49; PENDINGTIMESTAMP=51
  DELAYTIMER=52; CHANNELMASK=53; COUNT=54; PERIOD=55; SCAN_DURATION=56; ENERGY_LIST=57
  THREAD_DOMAIN_NAME=59; WAKEUP_CHANNEL=74; DISCOVERYREQUEST=128; DISCOVERYRESPONSE=129
  JOINERADVERTISEMENT=241
}

# reverse map: int -> name
$TN = @{}
foreach ($k in $T.Keys) { $TN[[int]$T[$k]] = [string]$k }

# clean + to bytes
$hex = ($Hex -replace '\s','').ToLower()
if ($hex.Length -eq 0 -or ($hex.Length % 2) -ne 0) { throw "Invalid hex input." }
[byte[]]$data = for ($i=0; $i -lt $hex.Length; $i+=2) { [Convert]::ToByte($hex.Substring($i,2),16) }

# decode
$pos = 0
while ($pos -lt $data.Length) {
  [byte]$tag = $data[$pos]; $pos++
  if ($pos -ge $data.Length) { throw "Truncated TLV after tag $tag." }
  [int]$len = $data[$pos]; $pos++
  if ($pos + $len -gt $data.Length) { throw "Truncated value for tag $tag (len $len)." }
  [byte[]]$val = $data[$pos..($pos+$len-1)]; $pos += $len

  $itag = [int]$tag
  $name = if ($TN.ContainsKey($itag)) { $TN[$itag] } else { "UNKNOWN" }

  if ($itag -eq 3) {
    $txt = [System.Text.Encoding]::ASCII.GetString($val)
    "t: {0,2} ({1}), l: {2}, v: {3}" -f $itag, $name, $len, $txt
  } else {
    $hexVal = (( $val | ForEach-Object { $_.ToString('x2') } ) -join '')
    "t: {0,2} ({1}), l: {2}, v: 0x{3}" -f $itag, $name, $len, $hexVal
  }
}
