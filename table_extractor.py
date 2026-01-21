#!/usr/bin/env python3
"""
Interface en ligne de commande pour le processeur de tableaux.

Ce script permet d'utiliser le processeur de tableaux refactorisé comme une commande
pour traiter des répertoires de PDFs ou d'images en mode batch.

Usage:
    python table_extractor.py <input_path> <output_path> [options]
    
Examples:
    # Traiter tous les PDFs d'un dossier
    python table_extractor.py ./pdfs ./output --extract-tables --refine-borders
    
    # Traiter un seul PDF
    python table_extractor.py './path_to/document.pdf' './path_to_output_folder/' --all --method line_detection -vv --export-json results.json
    
    # Traiter des images existantes
    python table_extractor.py './images' './output' --extract-tables --method contour
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any
import logging
import time
from datetime import datetime

# Import du code refactorisé
try:
    from extractor_module import (
        TableProcessor,
        ProcessingConfig,
        ExtractionMethod,
        ConsoleProgressObserver
    )
except ImportError as e:
    print(f"❌ Erreur d'importation: {e}")
    print("Assurez-vous que extractor_module.py est dans le même répertoire.")
    sys.exit(1)


class CLIProgressObserver(ConsoleProgressObserver):
    """Observer personnalisé pour l'interface en ligne de commande."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.verbose = verbose
        self.stats = {
            'stages_completed': 0,
            'total_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'start_time': time.time()
        }
    
    def on_stage_started(self, stage, message):
        msg = f"🔄 {stage.value}: {message}"
        if self.verbose > 1:
            print(msg)
        if self.verbose:
            logging.info(msg)
    
    def on_stage_completed(self, stage, message):
        self.stats['stages_completed'] += 1
        msg = f"✅ {stage.value}: {message}"
        if self.verbose > 1:
            print(msg)
        if self.verbose:
            logging.info(msg)
    
    def on_progress_update(self, current, total, message):
        percentage = (current / total) * 100 if total > 0 else 0
        msg = f"📊 Progression: {current}/{total} ({percentage:.1f}%) - {message}"
        if self.verbose > 1:
            print(msg)
        if self.verbose:
            logging.info(msg)
    
    def on_error(self, stage, error):
        msg = f"❌ ERREUR dans {stage.value}: {error}"
        if self.verbose > 1:
            print(msg)
        if self.verbose:
            logging.error(msg)
    
    def increment_file_stats(self, success: bool):
        self.stats['total_files'] += 1
        if success:
            self.stats['successful_files'] += 1
        else:
            self.stats['failed_files'] += 1
    
    def get_final_report(self) -> Dict[str, Any]:
        total_time = time.time() - self.stats['start_time']
        return {
            'total_time': f"{total_time:.2f}s",
            'stages_completed': self.stats['stages_completed'],
            'files_processed': self.stats['total_files'],
            'successful_files': self.stats['successful_files'],
            'failed_files': self.stats['failed_files'],
            'success_rate': (self.stats['successful_files'] / max(1, self.stats['total_files'])) * 100
        }


class TableCLI:
    """Interface en ligne de commande pour le processeur de tableaux."""
    
    def __init__(self):
        self.parser = self._create_parser()
        self.processor = None
        self.observer = None
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Créer le parser d'arguments de ligne de commande."""
        parser = argparse.ArgumentParser(
            description="Processeur de tableaux - Interface en ligne de commande",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Exemples d'utilisation:

  Traiter un PDF:
    python table_extractor.py document.pdf ./output
    
  Traiter un dossier de PDFs ou d'images:
    python table_extractor.py ./input_folder ./output
    
  Traiter une image:
    python table_extractor.py image.png ./output
    
  Mode verbose:
    python table_extractor.py ./input ./output --verbose
            """)
        
        # Arguments obligatoires
        parser.add_argument(
            'input_path',
            type=str,
            help='Chemin vers le fichier PDF, image ou répertoire contenant des PDFs/images'
        )
        
        parser.add_argument(
            'output_path',
            type=str,
            help='Répertoire de sortie pour les résultats'
        )
        
        # Option verbose uniquement
        parser.add_argument(
            '-v',
            '--verbose',
            action='count',
            default=0,
            help='Affichage détaillé du processus'
        )
        
        return parser
    
    def _setup_logging(self, output_dir: Path, input_name: str, verbose: bool = False):
        """Configurer le système de logging avec génération automatique du fichier de log."""
        log_level = logging.DEBUG if verbose else logging.INFO
        
        # Générer le nom du fichier de log avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{input_name}_{timestamp}.log"
        log_path = output_dir / log_filename
        
        # Effacer les handlers précédents pour éviter les doublons
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Configuration de base
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(log_level)
        
        # Format pour les logs
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Configuration du logger racine
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[file_handler],
            force=True
        )
        
        return log_path
    
    def _create_config(self) -> ProcessingConfig:
        """Créer la configuration de traitement par défaut."""
        return ProcessingConfig()
    
    def _collect_items(self, input_path: Path) -> List[Path]:
        """Collecter tous les éléments à traiter."""
        items = []
        
        if input_path.is_file():
            if input_path.suffix.lower() == '.pdf':
                items.append(input_path)
            elif input_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}:
                # Image individuelle
                items.append(input_path)
            else:
                logging.warning(f"Type de fichier non supporté: {input_path}")
        
        elif input_path.is_dir():
            # Rechercher les PDFs
            pdf_files = list(input_path.glob('*.pdf'))
            items.extend(pdf_files)
            
            # Si pas de PDFs, traiter le répertoire comme un répertoire d'images
            if not pdf_files:
                image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
                image_files = [
                    f for f in input_path.iterdir()
                    if f.is_file() and f.suffix.lower() in image_extensions
                ]
                
                if image_files:
                    # Traiter le répertoire entier comme une unité
                    items.append(input_path)
                else:
                    logging.warning(f"Aucun fichier PDF ou image trouvé dans: {input_path}")
        
        return items
    
    def _process_single_item(self, item_path: Path, output_base: Path) -> Dict[str, Any]:
        """Traiter un seul élément (PDF ou répertoire d'images)."""
        try:
            # Créer le processeur avec l'observateur
            processor = TableProcessor(observer=self.observer)
            
            # Options de traitement par défaut (tout activé)
            extract_images = True
            extract_tables = True
            refine_borders = True
            extraction_method = ExtractionMethod.LINE_DETECTION
            
            # Créer le répertoire de sortie spécifique
            if item_path.is_file():
                output_dir = output_base / item_path.stem
            else:
                output_dir = output_base / item_path.name
            
            # Traiter selon le type
            if item_path.is_file() and item_path.suffix.lower() == '.pdf':
                results = processor.process_pdf(
                    pdf_path=str(item_path),
                    output_dir=str(output_dir),
                    extract_images=extract_images,
                    extract_tables=extract_tables,
                    refine_borders=refine_borders,
                    extraction_method=extraction_method
                )
            elif item_path.is_file() and item_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}:
                # Image individuelle - créer un répertoire temporaire
                temp_image_dir = output_dir / "images"
                temp_image_dir.mkdir(parents=True, exist_ok=True)
                
                # Copier l'image dans le répertoire temporaire
                import shutil
                temp_image_path = temp_image_dir / item_path.name
                shutil.copy2(item_path, temp_image_path)
                
                results = processor.process_images(
                    images_dir=str(temp_image_dir),
                    output_dir=str(output_dir),
                    extract_tables=extract_tables,
                    refine_borders=refine_borders,
                    extraction_method=extraction_method
                )
            else:
                # Répertoire d'images
                results = processor.process_images(
                    images_dir=str(item_path),
                    output_dir=str(output_dir),
                    extract_tables=extract_tables,
                    refine_borders=refine_borders,
                    extraction_method=extraction_method
                )
            
            self.observer.increment_file_stats(success=True)
            return results
            
        except Exception as e:
            self.observer.increment_file_stats(success=False)
            error_msg = f"Erreur lors du traitement de {item_path}: {e}"
            logging.error(error_msg)
            raise  # Toujours propager l'erreur
            
            return {
                'error': str(e),
                'input_path': str(item_path),
                'success': False
            }
    

    
    def run(self, args=None):
        """Exécuter l'interface en ligne de commande."""
        if args is None:
            args = self.parser.parse_args()
        
        # Chemins d'entrée et de sortie
        input_path = Path(args.input_path).resolve()
        output_path = Path(args.output_path).resolve()
        
        if not input_path.exists():
            self.parser.error(f"Chemin d'entrée inexistant: {input_path}")
        
        # Créer le répertoire de sortie
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Configuration du logging avec génération automatique
        input_name = input_path.stem if input_path.is_file() else input_path.name
        log_path = self._setup_logging(output_path, input_name, args.verbose)
        
        # Configuration et initialisation
        config = self._create_config()
        self.observer = CLIProgressObserver(verbose=args.verbose)
        
        # Créer le processeur avec observer personnalisé et verbose correct
        from extractor_module import TableSegmentationPipeline
        pipeline = TableSegmentationPipeline(config, self.observer)
        self.processor = TableProcessor(config=config, verbose=args.verbose)
        self.processor._pipeline = pipeline  # Remplacer le pipeline par défaut
        
        if args.verbose:
            print(f"🚀 Démarrage du traitement")
            print(f"📁 Entrée: {input_path}")
            print(f"📁 Sortie: {output_path}")
            print(f"📄 Log: {log_path}")
        
        # Logger les informations principales
        logging.info(f"Démarrage du traitement")
        logging.info(f"Entrée: {input_path}")
        logging.info(f"Sortie: {output_path}")
        logging.info(f"Mode verbose: {args.verbose}")
        
        try:
            # Trouver les fichiers à traiter
            items_to_process = self._collect_items(input_path)
            
            if not items_to_process:
                print("⚠️ Aucun fichier à traiter trouvé")
                return 1
            
            if args.verbose:
                print(f"📋 {len(items_to_process)} élément(s) à traiter")
            logging.info(f"{len(items_to_process)} élément(s) à traiter")
            
            # Traiter tous les éléments
            all_results = []
            
            for i, item in enumerate(items_to_process, 1):
                msg = f"Traitement {i}/{len(items_to_process)}: {item.name}"
                if args.verbose:
                    print(f"\n📄 {msg}")
                logging.info(msg)
                
                try:
                    results = self._process_single_item(item, output_path)
                    all_results.append(results)
                    
                    summary = results.get('summary', {})
                    success_rate = summary.get('success_rate', 0)
                    total_files = summary.get('total_processed_files', 0)
                    success_msg = f"Terminé: {total_files} fichier(s), {success_rate:.1f}% de réussite"
                    if args.verbose:
                        print(f"  ✅ {success_msg}")
                    logging.info(success_msg)
                
                except Exception as e:
                    all_results.append({
                        'error': str(e),
                        'input_path': str(item),
                        'success': False
                    })
                    error_msg = f"Échec: {e}"
                    print(f"  ❌ {error_msg}")
                    logging.error(error_msg)
                    # Continue avec le prochain fichier
            
            # Rapport final
            final_report = self.observer.get_final_report()
            
            if args.verbose:
                print(f"\n{'='*50}")
                print(f"📊 RAPPORT FINAL")
                print(f"{'='*50}")
                print(f"⏱️  Temps total: {final_report['total_time']}")
                print(f"📁 Éléments traités: {final_report['files_processed']}")
                print(f"✅ Succès: {final_report['successful_files']}")
                print(f"❌ Échecs: {final_report['failed_files']}")
                print(f"📈 Taux de réussite: {final_report['success_rate']:.1f}%")
                print(f"📄 Log sauvegardé: {log_path}")
            
            # Logger le rapport final
            logging.info("RAPPORT FINAL")
            logging.info(f"Temps total: {final_report['total_time']}")
            logging.info(f"Éléments traités: {final_report['files_processed']}")
            logging.info(f"Succès: {final_report['successful_files']}")
            logging.info(f"Échecs: {final_report['failed_files']}")
            logging.info(f"Taux de réussite: {final_report['success_rate']:.1f}%")
            
            # Code de sortie basé sur le succès
            return 0 if final_report['failed_files'] == 0 else 1
            
        except KeyboardInterrupt:
            print("\n⏹️ Traitement interrompu par l'utilisateur")
            return 130
        
        except Exception as e:
            logging.error(f"Erreur fatale: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1


def main():
    """Point d'entrée principal."""
    cli = TableCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
