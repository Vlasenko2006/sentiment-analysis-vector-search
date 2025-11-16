#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 21:39:13 2025

@author: andreyvlasenko
"""
import os
import time
import shutil


def cleanup_old_jobs(logger, max_age_days=7):
    """
    Remove job folders older than max_age_days, but keep the root visualizations folder
    
    Args:
        max_age_days: Maximum age in days before job folders are deleted
    """
    try:
        base_dir = "my_volume/sentiment_analysis"
        if not os.path.exists(base_dir):
            return
        
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            
            # Skip the root visualizations folder
            if item == "visualizations":
                continue
            
            # Skip .DS_Store and other hidden files
            if item.startswith("."):
                continue
            
            # Only process directories (job folders are UUID-named directories)
            if os.path.isdir(item_path):
                # Check folder age
                folder_age = current_time - os.path.getmtime(item_path)
                
                if folder_age > max_age_seconds:
                    try:
                        shutil.rmtree(item_path)
                        logger.info(f"Cleanup: Removed old job folder: {item} (age: {folder_age/86400:.1f} days)")
                    except Exception as e:
                        logger.warning(f"Cleanup: Could not remove {item}: {e}")
    
    except Exception as e:
        logger.warning(f"Cleanup: Error during cleanup: {e}")