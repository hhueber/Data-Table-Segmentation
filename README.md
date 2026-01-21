# Table Processor - Extracteur de Tableaux PDF/Images

## 📋 Description

Extrait les pages des PDF en images, segmente et extrait les tableaux englobants, avec raffinement des bordures.

## 🎯 Fonctionnalités

- Sources possibles : un pdf, plusieurs pdfs dans un dossier cible, une image, plusieurs images dans un dossier cible
- Extraction des pages en images depuis des documents PDF
- Segmentation et extraction d'un tableau englobant depuis les images
- Raffinement de la bordure
- Production d'un log des opérations effectuées dans le dossier de sortie

## 📦 Dépendances

- **Python** : 3.12
- **deskew** : 1.5
- **NumPy** : 2.0
- **OpenCV** : 4.12
- **Pillow** : 11.3
- **PyMuPDF** : 1.26

## 🚀 Utilisation

Instructions d'installation et d'utilisation de l'outil Table Processor.

1. Créez un environnement virtuel, et activez-le :
   ```bash
   python -m venv env ; source env/bin/activate
   ```

2. Installez les dépendances via pip :
   ```bash
   pip install -r requirements.txt
   ```

3. Exécutez le script principal avec les arguments nécessaires :
   ```bash
    python table_extractor.py <chemin_vers_le_fichier> <chemin_de_sortie>
    ```

Options disponibles :

    Gestion d'une seule image :
    ```bash
    python table_extractor.py <image>.png <chemin_de_sortie>
    ```

    Gestion d'un seul pdf :
    ```bash
    python table_extractor.py <pdf>.pdf <chemin_de_sortie>
    ```
    
    Gestion de set de pdfs ou d'images dans un dossier cible :
    ```bash
    python table_extractor.py <dossier_cible> <chemin_de_sortie>
    ```

## 🛠️ Contributeurs

- Samy Ibriz Pelaez 
- Hugo Hueber
- Marion Rivoal

## 📄 Contact

Pour toute question ou suggestion, veuillez contacter [helpdesk@unil.ch](mailto:helpdesk@unil.ch?subject=%5BDCSR%5D%20Data%20Table%20Segmentation) en précisant "DCSR" dans l'objet du courriel.
