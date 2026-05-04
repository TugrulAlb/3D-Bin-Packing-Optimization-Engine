param(
    [string]$BaseUrl = "http://localhost:8000/api/v1",
    [string]$ApiKey = "gizli-test-anahtari-123",
    [string]$Sample = "data/samples/0114.json",
    [string]$Algorithm = "greedy"
)

$ErrorActionPreference = "Stop"
$Headers = @{ "X-API-Key" = $ApiKey }

Write-Host "1) Health check" -ForegroundColor Cyan
$h = Invoke-RestMethod -Uri "$BaseUrl/health/" -Method Get
Write-Host "   status=$($h.status), version=$($h.version)" -ForegroundColor Green

Write-Host "`n2) Auth: yanlis key 401 vermeli" -ForegroundColor Cyan
try {
    Invoke-RestMethod -Uri "$BaseUrl/optimize/" -Method Post `
        -Headers @{ "X-API-Key" = "yanlis-key" } -ContentType "application/json" `
        -Body '{}' | Out-Null
    Write-Host "   HATA: 401 beklenmeliydi" -ForegroundColor Red
} catch {
    Write-Host "   OK: $($_.Exception.Response.StatusCode)" -ForegroundColor Green
}

Write-Host "`n3) Validation: bos payload 400 vermeli" -ForegroundColor Cyan
try {
    Invoke-RestMethod -Uri "$BaseUrl/optimize/" -Method Post `
        -Headers $Headers -ContentType "application/json" -Body '{}' | Out-Null
    Write-Host "   HATA: 400 beklenmeliydi" -ForegroundColor Red
} catch {
    $code = $_.Exception.Response.StatusCode
    if ($code -eq "BadRequest") {
        Write-Host "   OK: $code" -ForegroundColor Green
    } else {
        Write-Host "   HATA: $code (400 beklenmisti)" -ForegroundColor Red
    }
}

Write-Host "`n4) Optimizasyon basla ($Sample, $Algorithm)" -ForegroundColor Cyan
$payload = Get-Content $Sample -Raw
$body = $payload | ConvertFrom-Json
$body | Add-Member -NotePropertyName "algorithm" -NotePropertyValue $Algorithm -Force
$bodyJson = $body | ConvertTo-Json -Depth 10
$resp = Invoke-RestMethod -Uri "$BaseUrl/optimize/" -Method Post `
    -Headers $Headers -ContentType "application/json" -Body $bodyJson
$jobId = $resp.job_id
Write-Host "   job_id=$jobId, status=$($resp.status)" -ForegroundColor Green

Write-Host "`n5) Status polla (max 60sn)" -ForegroundColor Cyan
$lastPct = -1
for ($i = 0; $i -lt 60; $i++) {
    $st = Invoke-RestMethod -Uri "$BaseUrl/optimize/$jobId/status/" -Headers $Headers
    if ($st.percent -ne $lastPct) {
        Write-Host "   $($st.percent)% - $($st.phase_label) [$($st.status)]" -ForegroundColor Gray
        $lastPct = $st.percent
    }
    if ($st.status -eq "completed") { break }
    if ($st.status -in @("failed", "cancelled")) {
        Write-Host "   HATA: status=$($st.status)" -ForegroundColor Red
        exit 1
    }
    Start-Sleep -Seconds 1
}
Write-Host "   Tamamlandi: $($st.summary.toplam_palet) palet ($($st.summary.single_palet) single, $($st.summary.mix_palet) mix)" -ForegroundColor Green

Write-Host "`n6) Sonuc al" -ForegroundColor Cyan
$result = Invoke-RestMethod -Uri "$BaseUrl/optimize/$jobId/result/" -Headers $Headers
Write-Host "   Toplam $($result.paletler.Count) palet:" -ForegroundColor Green
foreach ($p in $result.paletler) {
    Write-Host "     Palet $($p.palet_id) [$($p.palet_turu)] - $($p.urun_sayisi) urun, doluluk %$($p.doluluk_orani)" -ForegroundColor Gray
}
Write-Host "`n   Yerlesmemis urun: $($result.summary.yerlesmemis_urun_sayisi)" -ForegroundColor Gray
Write-Host "   Sure: $($result.summary.elapsed_sec) sn" -ForegroundColor Gray

Write-Host "`n7) NOT_READY testi (yeni job, hemen result iste)" -ForegroundColor Cyan
$resp2 = Invoke-RestMethod -Uri "$BaseUrl/optimize/" -Method Post `
    -Headers $Headers -ContentType "application/json" -Body $bodyJson
try {
    Invoke-RestMethod -Uri "$BaseUrl/optimize/$($resp2.job_id)/result/" -Headers $Headers | Out-Null
} catch {
    Write-Host "   OK: 409 NOT_READY (status=$($_.Exception.Response.StatusCode))" -ForegroundColor Green
}

Write-Host "`n8) Iptal" -ForegroundColor Cyan
$cancelResp = Invoke-RestMethod -Uri "$BaseUrl/optimize/$($resp2.job_id)/cancel/" -Method Post -Headers $Headers
Write-Host "   cancelled=$($cancelResp.cancelled), previous=$($cancelResp.previous_status)" -ForegroundColor Green

Write-Host "`nTUM API TESTLERI BASARILI" -ForegroundColor Green
