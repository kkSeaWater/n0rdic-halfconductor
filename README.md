

##risk.ps1
Monitors a Nordic Semiconductor device via serial port for 300 seconds by default. Logs some telemetry details (with "risk" algorithm):
- State
- Parent information (rloc16 of the parent)
- MAC counters
- Link quality index
- RTT
- Txt
And then composes a csv file with these data

- logger is tied 1:1 to openthread's internal state machine
- in order to use the power profiler kit II with the DK, cut the SB40 bridge on the DK, then power the DK using the PPKII on Ammeter mode
