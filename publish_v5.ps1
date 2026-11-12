param([switch]$PushOnly)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $repoRoot

if ($PushOnly) {
    & git push -u origin main
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n上传未完成。请检查上方 GitHub 登录提示、远端仓库地址和账号权限。" -ForegroundColor Red
    } else {
        Write-Host "`n上传完成。" -ForegroundColor Green
    }
    Read-Host '按 Enter 关闭此窗口'
    exit $LASTEXITCODE
}

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()

$form = New-Object System.Windows.Forms.Form
$form.Text = 'Learnlysis v5 · GitHub 上传'
$form.StartPosition = 'CenterScreen'
$form.Size = New-Object System.Drawing.Size(660, 540)
$form.MinimumSize = New-Object System.Drawing.Size(660, 540)
$form.Font = New-Object System.Drawing.Font('Microsoft YaHei UI', 9)

function Add-Label($text, $x, $y, $width = 150) {
    $label = New-Object System.Windows.Forms.Label
    $label.Text = $text
    $label.Location = New-Object System.Drawing.Point($x, $y)
    $label.Size = New-Object System.Drawing.Size($width, 25)
    $form.Controls.Add($label)
    return $label
}

function Add-TextBox($x, $y, $width, $value = '') {
    $box = New-Object System.Windows.Forms.TextBox
    $box.Location = New-Object System.Drawing.Point($x, $y)
    $box.Size = New-Object System.Drawing.Size($width, 25)
    $box.Text = $value
    $form.Controls.Add($box)
    return $box
}

Add-Label 'GitHub 账号（仓库所有者）' 20 20 190 | Out-Null
$ownerBox = Add-TextBox 220 18 390
Add-Label '仓库名称' 20 58 190 | Out-Null
$repoBox = Add-TextBox 220 56 390 'Learnlysis-PySide6-v5'
Add-Label '提交署名' 20 96 190 | Out-Null
$nameBox = Add-TextBox 220 94 390
Add-Label '提交邮箱（建议 GitHub noreply 邮箱）' 20 134 190 | Out-Null
$emailBox = Add-TextBox 220 132 390
Add-Label '提交说明' 20 172 190 | Out-Null
$messageBox = Add-TextBox 220 170 390 '发布 PySide6 v5'

$accountNote = New-Object System.Windows.Forms.Label
$accountNote.Text = '注意：上面的署名只用于 Git 提交记录。GitHub 实际登录仍由 Git Credential Manager 处理；首次推送若显示旧账号，请先清除 Windows 凭据管理器中的 github.com 凭据，再登录目标账号。此窗口不会保存密码或 Token。'
$accountNote.Location = New-Object System.Drawing.Point(20, 210)
$accountNote.Size = New-Object System.Drawing.Size(600, 64)
$accountNote.ForeColor = [System.Drawing.Color]::FromArgb(75, 75, 85)
$form.Controls.Add($accountNote)

$newRepoButton = New-Object System.Windows.Forms.Button
$newRepoButton.Text = '打开 GitHub 新建仓库'
$newRepoButton.Location = New-Object System.Drawing.Point(20, 282)
$newRepoButton.Size = New-Object System.Drawing.Size(190, 34)
$newRepoButton.Add_Click({ Start-Process 'https://github.com/new' })
$form.Controls.Add($newRepoButton)

$credentialButton = New-Object System.Windows.Forms.Button
$credentialButton.Text = '打开凭据管理器'
$credentialButton.Location = New-Object System.Drawing.Point(220, 282)
$credentialButton.Size = New-Object System.Drawing.Size(170, 34)
$credentialButton.Add_Click({ Start-Process 'control.exe' -ArgumentList '/name Microsoft.CredentialManager' })
$form.Controls.Add($credentialButton)

$uploadButton = New-Object System.Windows.Forms.Button
$uploadButton.Text = '准备提交并上传'
$uploadButton.Location = New-Object System.Drawing.Point(445, 282)
$uploadButton.Size = New-Object System.Drawing.Size(165, 34)
$uploadButton.BackColor = [System.Drawing.Color]::FromArgb(93, 76, 235)
$uploadButton.ForeColor = [System.Drawing.Color]::White
$form.Controls.Add($uploadButton)

$logBox = New-Object System.Windows.Forms.TextBox
$logBox.Location = New-Object System.Drawing.Point(20, 330)
$logBox.Size = New-Object System.Drawing.Size(590, 155)
$logBox.Multiline = $true
$logBox.ReadOnly = $true
$logBox.ScrollBars = 'Vertical'
$logBox.Text = "仓库目录：$repoRoot`r`n上传范围由 .gitignore 限定为 PySide6 v5、发布包和必要许可文件。`r`n第一次使用：先在 GitHub 建立空仓库，再填写账号、仓库名及提交署名。"
$form.Controls.Add($logBox)

$uploadButton.Add_Click({
    $owner = $ownerBox.Text.Trim()
    $repo = $repoBox.Text.Trim()
    $author = $nameBox.Text.Trim()
    $email = $emailBox.Text.Trim()
    $message = $messageBox.Text.Trim()
    if (-not $owner -or -not $repo -or -not $author -or -not $email) {
        [System.Windows.Forms.MessageBox]::Show('请填写 GitHub 账号、仓库名、提交署名和邮箱。', '信息不完整', 'OK', 'Warning') | Out-Null
        return
    }
    if ($owner -notmatch '^[A-Za-z0-9-]+$' -or $repo -notmatch '^[A-Za-z0-9_.-]+$') {
        [System.Windows.Forms.MessageBox]::Show('账号或仓库名格式不正确。', '无法继续', 'OK', 'Warning') | Out-Null
        return
    }
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        [System.Windows.Forms.MessageBox]::Show('没有找到 Git。请先安装 Git for Windows，再重新打开本程序。', '缺少 Git', 'OK', 'Error') | Out-Null
        return
    }

    $uploadButton.Enabled = $false
    $locationPushed = $false
    try {
        Push-Location -LiteralPath $repoRoot
        $locationPushed = $true
        if (-not (Test-Path (Join-Path $repoRoot '.git'))) {
            & git init -b main 2>&1 | ForEach-Object { $logBox.AppendText("`r`n$_") }
            if ($LASTEXITCODE -ne 0) { throw '初始化本地仓库失败。' }
        }
        & git branch -M main 2>&1 | ForEach-Object { $logBox.AppendText("`r`n$_") }
        if ($LASTEXITCODE -ne 0) { throw '设置 main 分支失败。' }
        & git config --local user.name $author
        if ($LASTEXITCODE -ne 0) { throw '设置本地提交署名失败。' }
        & git config --local user.email $email
        if ($LASTEXITCODE -ne 0) { throw '设置本地提交邮箱失败。' }

        $remoteUrl = "https://github.com/$owner/$repo.git"
        & git remote get-url origin *> $null
        if ($LASTEXITCODE -eq 0) {
            & git remote set-url origin $remoteUrl
        } else {
            & git remote add origin $remoteUrl
        }
        if ($LASTEXITCODE -ne 0) { throw '设置 GitHub 远端地址失败。' }

        & git add -A
        if ($LASTEXITCODE -ne 0) { throw '整理上传文件失败。' }
        & git diff --cached --quiet
        $hasChanges = ($LASTEXITCODE -eq 1)
        if ($LASTEXITCODE -gt 1) { throw '检查待提交文件失败。' }
        if ($hasChanges) {
            if (-not $message) { $message = '更新 PySide6 v5' }
            & git commit -m $message 2>&1 | ForEach-Object { $logBox.AppendText("`r`n$_") }
            if ($LASTEXITCODE -ne 0) { throw '创建 Git 提交失败。' }
        } else {
            $logBox.AppendText("`r`n本地文件没有新改动，将直接推送现有提交。")
        }
        Pop-Location
        $locationPushed = $false

        $logBox.AppendText("`r`n`r`n远端已设为 $remoteUrl。即将打开终端执行推送；请按 Git Credential Manager 提示登录目标账号。")
        $psArgs = @('-NoExit', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"", '-PushOnly')
        Start-Process -FilePath (Join-Path $PSHOME 'powershell.exe') -ArgumentList $psArgs -WorkingDirectory $repoRoot
    } catch {
        if ($locationPushed) { Pop-Location -ErrorAction SilentlyContinue }
        $logBox.AppendText("`r`n错误：$($_.Exception.Message)")
        [System.Windows.Forms.MessageBox]::Show($_.Exception.Message, '准备上传失败', 'OK', 'Error') | Out-Null
    } finally {
        $uploadButton.Enabled = $true
    }
})

[void]$form.ShowDialog()
