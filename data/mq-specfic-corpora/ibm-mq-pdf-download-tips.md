## 📄 IBM MQ PDF Documentation — Versions 9.2 → 9.4+

| Version        | What you get                                                                                                                 | Link (PDF documentation / download page)                                                                                                                                                       |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **IBM MQ 9.2** | Full product documentation (install, admin, config, secure, reference, z/OS Program Directories, etc.)                       | [IBM MQ 9.2 PDF product documentation & Program Directories](https://www.ibm.com/docs/SSFKSJ_9.2.0/com.ibm.mq.pro.doc/q001040_.htm)                                                 |
| **IBM MQ 9.3** | Full product documentation + z/OS Program Directories in PDF format                                                          | [IBM MQ 9.3 PDF files for product documentation & Program Directories](https://www.ibm.com/docs/en/ibm-mq/9.3.x?topic=am-mq-93-pdf-files-product-documentation-program-directories) |
| **IBM MQ 9.4** | Full documentation bundle: install, admin, config, secure, container docs, z/OS Program Directories, reference manuals, etc. | [IBM MQ 9.4 PDF product documentation & Program Directories](https://www.ibm.com/docs/en/ibm-mq/9.4.x?topic=am-mq-94-pdf-files-product-documentation-program-directories)            |


### ℹ️ Notes on Using These Links

* For each version, IBM supplies a set of PDFs that correspond to all the major documentation sections (overview, installation, admin, configuration, security, reference, z/OS program directories, etc.). ([IBM][1])
* For “older versions” (e.g., 9.2 and 9.3) — the official documentation site provides an **archive/legacy** navigation via “Older versions” so you can still access and download the docs even if a version is not the latest. ([IBM][4])
* For 9.2, there is a publicly accessible PDF bundle, including full docs and z/OS Program Directories. ([IBM][1])
* For 9.3 and 9.4, you often get both “Multiplatforms” and “z/OS / Program Directories” documentation sets. ([IBM][2])

---
## **Command-line Download Examples** (Linux and Windows)

## ✅ URL to download

* `https://public.dhe.ibm.com/software/integration/wmq/docs/V9.4/PDFs/mq94.install.pdf`

---

## 🐧 On Linux / macOS (bash, WSL, etc.)

Using `curl` (standard tool on most Unix-like systems):

```bash
curl -L -O https://public.dhe.ibm.com/software/integration/wmq/docs/V9.4/PDFs/mq94.install.pdf
```

* `-L` tells curl to follow redirects (useful in some cases). ([Curl][1])
* `-O` tells curl to save the file using the remote name (`mq94.install.pdf`). ([TechTarget][2])

Alternatively, using `wget`:

```bash
wget https://public.dhe.ibm.com/software/integration/wmq/docs/V9.4/PDFs/mq94.install.pdf
```

If you want to force the output directory or filename:

```bash
wget -O ~/Downloads/mq94.install.pdf https://public.dhe.ibm.com/software/integration/wmq/docs/V9.4/PDFs/mq94.install.pdf
```

---

## 🪟 On Windows (PowerShell)

Using built-in PowerShell `Invoke-WebRequest`:

```powershell
Invoke-WebRequest -Uri "https://public.dhe.ibm.com/software/integration/wmq/docs/V9.4/PDFs/mq94.install.pdf" -OutFile "mq94.install.pdf"
```

This downloads the PDF and saves it as `mq94.install.pdf` in the current directory. ([ITPro Today][3])

If you prefer `curl.exe` (on recent Windows with curl available):

```powershell
curl.exe -L -o mq94.install.pdf https://public.dhe.ibm.com/software/integration/wmq/docs/V9.4/PDFs/mq94.install.pdf
```

The `-L` flag follows redirects; `-o` sets the output filename. ([Microsoft Learn][4])

---

## 📂 Example with your project structure

If you want to download into your `docs/ibm-mq-pdfs/` directory (inside your repo) on Linux/macOS:

```bash
mkdir -p docs/ibm-mq-pdfs
cd docs/ibm-mq-pdfs
curl -L -O https://public.dhe.ibm.com/software/integration/wmq/docs/V9.4/PDFs/mq94.install.pdf
```

On Windows (PowerShell), from the root of your repo:

```powershell
New-Item -ItemType Directory -Force -Path .\docs\ibm-mq-pdfs
cd .\docs\ibm-mq-pdfs
Invoke-WebRequest -Uri "https://public.dhe.ibm.com/software/integration/wmq/docs/V9.4/PDFs/mq94.install.pdf" -OutFile "mq94.install.pdf"
```
