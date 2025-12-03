## 📄 IBM MQ PDF Documentation — Versions 9.2 → 9.4+

| Version                          | Description / What you get                                                                                                                                                   | Link to PDF documentation / download page                                         |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **IBM MQ 9.2**                   | Full product documentation (core guides: install, admin, config, reference, security, developer, etc.) + z/OS Program Directories                                            | IBM MQ 9.2 PDF files for product documentation and Program Directories ([IBM][1]) |
| **IBM MQ 9.3**                   | Full product documentation in PDF form — quick-start guide, admin, secure, install, reference, container guide, z/OS directories, etc. ([IBM][2])                            | IBM MQ 9.3 PDF files for product documentation & Program Directories ([IBM][2])   |
| **IBM MQ 9.4 (Latest LTS / CD)** | Full documentation bundle: README, Quick Start Guide, install, plan, secure, admin, monitor, developer, reference, container docs, z/OS Program Directories, etc. ([IBM][3]) | IBM MQ 9.4 PDF files for product documentation & Program Directories ([IBM][3])   |

### ℹ️ Notes on Using These Links

* For each version, IBM supplies a set of PDFs that correspond to all the major documentation sections (overview, installation, admin, configuration, security, reference, z/OS program directories, etc.). ([IBM][1])
* For “older versions” (e.g., 9.2 and 9.3) — the official documentation site provides an **archive/legacy** navigation via “Older versions” so you can still access and download the docs even if a version is not the latest. ([IBM][4])
* For 9.2, there is a publicly accessible PDF bundle, including full docs and z/OS Program Directories. ([IBM][1])
* For 9.3 and 9.4, you often get both “Multiplatforms” and “z/OS / Program Directories” documentation sets. ([IBM][2])

---

## ✅ Recommendations — Tips for Downloading & Archiving

* **Download the entire PDF bundle** (all PDFs in the same folder) so that in-PDF hyperlinks between sections remain valid.
* **Mirror or store locally** (on your mainframe or internal git repo) so you have documentation even when offline or after IBM deprecates older versions.
* **For z/OS deployments**: be sure to fetch the “Program Directory” PDFs — they contain mainframe-specific info, datasets, macro reference, SMF formats, etc.
* **When working across versions**: keep separate folder trees (e.g., `docs/9.2/`, `docs/9.3/`, `docs/9.4/`) so you don’t mix version-specific behaviors or commands.
* **If you want offline browsing + search**: consider using the “IBM Documentation Offline” app + the corresponding MQ doc package per version. ([IBM][4])
