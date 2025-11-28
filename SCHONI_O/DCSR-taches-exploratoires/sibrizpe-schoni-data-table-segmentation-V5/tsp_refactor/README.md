# Table Processor - Extracteur de Tableaux PDF/Images

## 📋 Description

Extrait les pages des PDF en images, segmente et extrait les tableaux englobants, avec raffinement des bordures.

## 🎯 Fonctionnalitées
- Sources possibles : un pdf, plusieurs pdfs dans un dossier cible, une image, plusieurs images dans un dossier cible
- Extraction des pages en images depuis des documents PDF
- Segmentation et extraction d'un tableau englobant depuis les images
- Raffinement de la bordure
- Production d'un log des opérations effectuées dans le dossier de sortie

## 📦 Dépendances
- **Python** : 3.13.1
- **OpenCV** : 4.11
- **Pillow** : 11.2.1
- **PyMuPDF** : 1.26.0
- **NumPy** : 2.2.6

## 🚀 Utilisation

Instructions d'installation et d'utilisation de l'outil Table Processor.

1. Installez les dépendances via pip :
   ```bash
   pip install -r requirements.txt
   ```

2. Exécutez le script principal avec les arguments nécessaires :
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

Pour toute question ou suggestion, veuillez contacter helpdesk@unil.ch