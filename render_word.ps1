param([Parameter(Mandatory=$true)][string]$InputDocx,[Parameter(Mandatory=$true)][string]$OutputPdf)
$ErrorActionPreference='Stop'
$word=$null
$doc=$null
try {
    $word=New-Object -ComObject Word.Application
    $word.Visible=$false
    $word.DisplayAlerts=0
    $word.AutomationSecurity=3
    $doc=$word.Documents.Open((Resolve-Path -LiteralPath $InputDocx).Path,$false,$true)
    $doc.Repaginate()
    $doc.ExportAsFixedFormat([System.IO.Path]::GetFullPath($OutputPdf),17)
    Write-Output ('pages='+$doc.ComputeStatistics(2))
} finally {
    if ($null -ne $doc) {$doc.Close(0); [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc)}
    if ($null -ne $word) {$word.Quit(); [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)}
}
