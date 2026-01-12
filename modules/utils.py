import os
import sys
import shutil
import time
import logging
from pathlib import Path
from config import LOG_FILE

def setup_logger(name="NS-DMN"):
    """Configures a thread-safe logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # File Handler
        fh = logging.FileHandler(LOG_FILE)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
        
        # Console Handler
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(sh)
        
    return logger

logger = setup_logger("Utils")

def atomic_dir_swap(target_dir: Path, new_dir: Path, retries: int = 5):
    """
    Windows-safe atomic directory swap.
    Strategy:
    1. Rename target -> target.bak
    2. Rename new -> target
    3. Delete target.bak
    """
    target_dir = Path(target_dir)
    new_dir = Path(new_dir)
    backup_dir = target_dir.with_suffix(".bak")

    # 0. Safety Check
    if not new_dir.exists():
        logger.error(f"Swap Failed: Source {new_dir} does not exist.")
        return False

    # 1. Clear old backup if it exists (Lazy Cleanup)
    if backup_dir.exists():
        try:
            shutil.rmtree(backup_dir)
        except OSError as e:
            logger.warning(f"Could not clear old backup {backup_dir}: {e}")
            return False

    # 2. Rename Target -> Backup
    if target_dir.exists():
        success = False
        for i in range(retries):
            try:
                os.rename(target_dir, backup_dir)
                success = True
                break
            except OSError as e:
                logger.warning(f"Retry {i+1}/{retries} renaming target to backup: {e}")
                time.sleep(0.2)
        
        if not success:
            logger.error("Failed to move current data to backup. Aborting swap.")
            # Cleanup NEW dir so we don't leave junk
            try:
                shutil.rmtree(new_dir)
            except:
                pass
            return False

    # 3. Rename New -> Target
    try:
        os.rename(new_dir, target_dir)
    except OSError as e:
        logger.critical(f"SWAP FAILURE! Moved target to backup but failed to move new to target: {e}. Attempting Rollback.")
        # ROLLBACK
        try:
            os.rename(backup_dir, target_dir)
            logger.info("Rollback successful.")
        except OSError as e2:
            logger.critical(f"FATAL: Rollback FAILED. Data is in {backup_dir}. Error: {e2}")
        return False

    # 4. Success - Delete Backup
    try:
        shutil.rmtree(backup_dir)
    except OSError as e:
        logger.warning(f"Swap successful, but failed to delete backup {backup_dir}: {e}")
    
    logger.info("Atomic Directory Swap Completed Successfully.")
    return True

def recover_previous_state(target_dir: Path):
    """
    Checks for a .bak directory on startup. If target is missing but bak exists,
    it implies a crash during the previous swap (Step 3 or 4).
    Restores .bak to target.
    """
    target_dir = Path(target_dir)
    backup_dir = target_dir.with_suffix(".bak")
    
    # startup logic
    if not target_dir.exists() and backup_dir.exists():
        logger.warning("Startup Recovery: Found orphaned backup but no main data. Restoring...")
        try:
            os.rename(backup_dir, target_dir)
            logger.info("Recovery successful: Backup restored to Main.")
        except OSError as e:
            logger.critical(f"Startup Recovery Failed: {e}")
