wget -O models.tar.gz https://uc8229d504b3bd392fecf1a45c23.dl.dropboxusercontent.com/cd/0/get/A5a-YuQRFADK8WNWuSUnlIH7CjszEQM2ImcAqxfamJ1CzoKycVwrOeIS5I56-qSmYtTydU6J9XJK45qpvAjfwCvLv7ATMu0NnuuKeKjdHeieuQ/file?_download_id=2435093372208747612376414003391472716553430118742297638204638257&_notify_domain=www.dropbox.com&dl=1
wget -O cad.h5 https://uc471df38f729b6c7a4108b1cd7b.dl.dropboxusercontent.com/cd/0/get/A5ZSzkWQShS1Q50OcwVSkfApvWkxMhJhEszzNoP0d4hGszipOKwJ-8LFfekMq2FZaZn4_hxFEu4pOlGSC_Ea9OxiPbFjPSPKwcTo8As8dyEH5Q/file?_download_id=76829492963176478061382995572584114896785278751919913557986726824&_notify_domain=www.dropbox.com&dl=1
wget -O synthetic.tar.gz https://ucebdd1faf36eb2e812019e72a9c.dl.dropboxusercontent.com/cd/0/get/A5ZcyEEcUM240Ql1hQbBWd0OuiybjxRiO1TkCKRyCD8w6aQddBW-0rz5KUM6fTfypjPUGQVM9yP1vUcbhDqDx9s5A8V9vh3xZJW_-P-sF5QYtA/file?_download_id=2370271348801931715593462739449658887804921709174812690442303609&_notify_domain=www.dropbox.com&dl=1
tar -zxvf synthetic.tar.gz -C data/
tar -zxvf models.tar.gz -C trained_models/
mv cad.h5 data/cad/
