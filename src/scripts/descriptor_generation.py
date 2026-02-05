import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from matminer.featurizers.composition import (
    ElementProperty, Stoichiometry, ValenceOrbital, 
    IonProperty, AtomicOrbitals, Meredig
)

from pymatgen.core import Composition
import logging
from tqdm import tqdm
import warnings

logger = logging.getLogger(__name__)


class DescriptorGenerator:
    """Generate comprehensive materials and molecular descriptors using matminer and RDKit."""
    
    def __init__(self):
        """Initialize featurizers."""
        # Composition featurizers - core
        self.elem_prop = ElementProperty.from_preset("magpie")
        self.stoich = Stoichiometry()
        self.meredig = Meredig()
        
        # Cache for successful featurizations
        self._composition_cache: Dict = {}
        
    def generate_molecular_descriptors(self, df: pd.DataFrame, 
                                      smiles_column: str = 'smiles') -> pd.DataFrame:
        """Generate molecular descriptors from SMILES strings.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            smiles_column (str): Column name containing SMILES strings. Defaults to 'smiles'.
            
        Returns:
            pd.DataFrame: Dataframe with added molecular descriptor columns.
        """
        if smiles_column not in df.columns:
            logger.warning(f"Column {smiles_column} not found")
            return df
            
        logger.info("Generating molecular descriptors...")
        df_enriched = df.copy()
        
        # Initialize descriptor columns
        descriptor_names = []
        for name, _ in Descriptors.descList:
            descriptor_names.append(f"feat_mol_{name}")
            df_enriched[f"feat_mol_{name}"] = np.nan
            
        # Morgan fingerprints columns
        n_bits = 2048
        for i in range(n_bits):
            df_enriched[f"feat_morgan_fp_{i}"] = 0
            
        # Process each molecule
        for idx, smiles in tqdm(df_enriched[smiles_column].items(), total=len(df_enriched)):
            if pd.isna(smiles):
                continue
                
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                    
                # Calculate descriptors
                for name, func in Descriptors.descList:
                    try:
                        value = func(mol)
                        df_enriched.at[idx, f"feat_mol_{name}"] = value
                    except:
                        pass
                        
                # Calculate Morgan fingerprints
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
                for i in range(n_bits):
                    df_enriched.at[idx, f"feat_morgan_fp_{i}"] = fp[i]
                    
            except Exception as e:
                logger.debug(f"Error processing SMILES {smiles}: {e}")
                
        return df_enriched
    
        
    def generate_composition_descriptors(self, df: pd.DataFrame,
                                        formula_column: str = 'formula') -> pd.DataFrame:
        """Generate comprehensive composition-based descriptors.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            formula_column (str): Column name containing chemical formulas. Defaults to 'formula'.
            
        Returns:
            pd.DataFrame: Dataframe with added composition descriptor columns.
        """
        if formula_column not in df.columns:
            logger.warning(f"Column {formula_column} not found")
            return df
            
        logger.info("Generating composition descriptors...")
        df_enriched = df.copy()
        
        # Convert formulas to Composition objects
        compositions = []
        valid_indices = []
        
        for idx, formula in df_enriched[formula_column].items():
            if pd.isna(formula):
                continue
            try:
                comp = Composition(formula)
                compositions.append(comp)
                valid_indices.append(idx)
            except Exception as e:
                logger.debug(f"Error parsing formula {formula}: {e}")
                
        if not compositions:
            logger.warning("No valid compositions found")
            return df_enriched
        
        # Featurizers to apply
        featurizers = [
            ("comp", self.elem_prop),
            ("stoich", self.stoich),
            ("meredig", self.meredig),
        ]
        
        for prefix, featurizer in featurizers:
            self._apply_composition_featurizer(
                df_enriched, compositions, valid_indices, featurizer, prefix
            )
                    
        return df_enriched
    
    def _apply_composition_featurizer(self, df_enriched: pd.DataFrame,
                                     compositions: List[Composition],
                                     valid_indices: List, featurizer,
                                     prefix: str) -> None:
        """Apply a composition featurizer and add features to dataframe.
        
        Args:
            df_enriched (pd.DataFrame): Dataframe to add features to.
            compositions (List[Composition]): List of pymatgen Composition objects.
            valid_indices (List): List of indices corresponding to valid compositions.
            featurizer: Matminer featurizer object with featurize_many method.
            prefix (str): Prefix to prepend to feature column names.
        """
        try:
            logger.info(f"Calculating {prefix} features...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                features = featurizer.featurize_many(compositions, ignore_errors=True)
                feature_names = featurizer.feature_labels()
            
            for i, idx in enumerate(valid_indices):
                if features[i] is not None:
                    for j, name in enumerate(feature_names):
                        try:
                            col_name = f"feat_{prefix}_{name}"
                            value = features[i][j]
                            # Convert to numeric, skip if problematic
                            if isinstance(value, (int, float, np.number)):
                                df_enriched.at[idx, col_name] = float(value)
                        except (ValueError, IndexError, TypeError) as e:
                            logger.debug(f"Error setting {prefix}_{name} at idx {idx}: {e}")
        except Exception as e:
            logger.warning(f"Failed to compute {prefix} features: {e}")
    
    
    def generate_all_descriptors(self, df: pd.DataFrame,
                                smiles_column: Optional[str] = 'smiles',
                                formula_column: Optional[str] = 'formula') -> pd.DataFrame:
        """Generate all available descriptors (molecular and composition-based).
        
        Args:
            df (pd.DataFrame): Input dataframe.
            smiles_column (Optional[str]): Column containing SMILES strings. Defaults to 'smiles'.
            formula_column (Optional[str]): Column containing chemical formulas. Defaults to 'formula'.
            
        Returns:
            pd.DataFrame: Dataframe with all generated descriptors.
        """
        df_result = df.copy()
        
        if smiles_column and smiles_column in df.columns:
            df_result = self.generate_molecular_descriptors(df_result, smiles_column)
        
        if formula_column and formula_column in df.columns:
            df_result = self.generate_composition_descriptors(df_result, formula_column)
        
        return df_result
    
    def get_feature_names(self, formula: Optional[str] = None) -> Dict[str, List[str]]:
        """Get feature names for composition and molecular descriptors.
        
        Args:
            formula (Optional[str]): Chemical formula to generate feature names for. If provided,
                validates that the formula can be parsed. If None, returns generic feature names.
            
        Returns:
            Dict[str, List[str]]: Dictionary with feature type keys and list of feature names as values.
                Composition features are only included if a valid formula is provided.
        """
        feature_names = {}
        
        # Molecular features
        mol_descriptors = [f"feat_mol_{name}" for name, _ in Descriptors.descList]
        feature_names['molecular'] = mol_descriptors
        
        morgan_fingerprints = [f"feat_morgan_fp_{i}" for i in range(2048)]
        feature_names['morgan_fingerprint'] = morgan_fingerprints
        
        # Composition features (only if formula provided and valid)
        if formula is not None:
            try:
                comp = Composition(formula)
                
                # Get feature names from each featurizer
                comp_features = [f"feat_comp_{name}" for name in self.elem_prop.feature_labels()]
                feature_names['element_property'] = comp_features
                
                stoich_features = [f"feat_stoich_{name}" for name in self.stoich.feature_labels()]
                feature_names['stoichiometry'] = stoich_features
                
                meredig_features = [f"feat_meredig_{name}" for name in self.meredig.feature_labels()]
                feature_names['meredig'] = meredig_features
                
            except Exception as e:
                logger.warning(f"Could not generate composition feature names for formula '{formula}': {e}")
        
        return feature_names
    
