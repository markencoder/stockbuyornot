param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CommandArgs
)

& "C:\ProgramData\Anaconda3\envs\tower312\Scripts\stockbuyornot.exe" @CommandArgs
