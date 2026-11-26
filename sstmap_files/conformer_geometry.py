#!/usr/bin/env python
import copy
import logging
import queue
from functools import partial
from itertools import combinations,permutations
import numpy as np
import pandas as pd
from rdkit import Chem,Geometry
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdDistGeom
from rdkit.Geometry import Point3D
from rdkit.Chem.rdMolTransforms import TransformConformer
from rdkit.Numerics import rdAlignment
from multiprocessing import Pool
from typing import Dict, Iterable, List, Set, Tuple
from scipy.spatial import distance_matrix

class ConformerRMSD:
    """Adapted version of rdMolAlign.GetBestAlignmentTransform
    """

    UNIQUIFY = False
    RECURSION_POSSIBLE = True
    USE_CHIRALITY = False
    USE_QUERY_QUERY_MATCHES = False

    def __init__(
        self,
        mol: Chem.Mol,
        max_matches: int = 1000000,
        weights: list = None,
        reflect: bool = False,
        max_iters: int = 50,
    ):

        if weights is None:
            weights = []

        self.mol = mol
        self.max_matches = max_matches
        self.weights = weights
        self.reflect = reflect
        self.max_iters = max_iters
        self.matches = self.find_matches()

    def find_matches(self) -> Tuple[Tuple[int]]:
        """find all symmetries
        """

        sub_match_ps = Chem.SubstructMatchParameters()
        sub_match_ps.uniquify = self.UNIQUIFY
        sub_match_ps.recursionPossible = self.RECURSION_POSSIBLE
        sub_match_ps.useChirality = self.USE_CHIRALITY
        sub_match_ps.useQueryQueryMatches = self.USE_QUERY_QUERY_MATCHES
        sub_match_ps.maxMatches = self.max_matches
        matches = self.mol.GetSubstructMatches(self.mol, sub_match_ps)
        if len(matches)>self.max_matches:
            name = self.prb_mol.GetProp("Name") if self.prb_mol.HasProp("Name") else "UNNAMED"
            _logger.warning("Warning in calculation of conformer RMSD:")
            _logger.warning(f"{len(matches)} matches detected for molecule {name}.")
        return matches
        
    def align_confs_on_atom_map(self, match: Tuple[int], conf_idx_1: int, conf_idx_2: int):
        """find minimal MSD for atoms in two conformers

        match (Tuple[int]): mapping of atoms in first conformer to atoms in second conformer
        conf_idx_1: index of first conformer
        conf_idx_2: index of second conformer
        
        Returns: MSD (float), trans (np.array) 4x4 matrix of transformation to fit conf_2 to 1
        """

        conf_1 = self.mol.GetConformer(conf_idx_1)
        conf_2 = self.mol.GetConformer(conf_idx_2)
        points_1 = [conf_1.GetAtomPosition(i) for i in range(len(match))]
        points_2 = [conf_2.GetAtomPosition(i) for i in match]
        ssr,trans = rdAlignment.GetAlignmentTransform(
            points_1, points_2, self.weights, self.reflect, self.max_iters
        )
        return ssr/len(points_1), trans

    def get_best_rmsd(self, conf_idx_1: int, conf_idx_2: int):

        """find minimal rmsd of two conformers, taking symmetry into account

        conf_idx_1: index of first conformer
        conf_idx_2: index of second conformer

        Returns: min_rmsd (float), best_trans (np.array) 4x4 matrix of transformation 
        to fit conf_2 to 1, best_match (Tuple[int]) best matching of atoms for optimal fit
        """

        assert self.matches
        best_msd = np.inf
        best_match = None
        best_trans = None
        for match in self.matches:
            msd, tmp_trans = self.align_confs_on_atom_map(match, conf_idx_1, conf_idx_2)
            if msd<best_msd:
                best_msd = msd
                best_match = match
                best_trans = tmp_trans
        assert best_match is not None
        assert best_trans is not None
        self.best_match = best_match
        self.best_trans = best_trans
        return np.sqrt(best_msd), best_trans, best_match

class HydrogenConformerRMSD:

    """Allow symmetries of molecules for RMSD calculation
       Avoiding combinatorial explosion due to hydrogens
    """

    def __init__(
        self,
        mol: Chem.Mol,
        max_matches: int = 1000000,
        weights: list = None,
        reflect: bool = False,
        max_iters: int = 50,
    ):

        if weights is None:
            weights = []

        # molecules
        self.mol = Chem.Mol(mol)
        self._mol_no_h = Chem.RemoveHs(mol)

        # match
        self._match_h_removed = self.mol.GetSubstructMatch(self._mol_no_h)

        # indices of conformer pairs, lazy initialization
        self._combinations = list()
        self._conf_idcs_to_list_idx = dict()

        # settings for RMSD
        self.max_matches = max_matches
        self.weights = weights
        self.reflect = reflect
        self.max_iters = max_iters

        # optimal matches heavy atoms
        self.conf_rmsd = ConformerRMSD(
            self._mol_no_h,
            max_matches = self.max_matches,
            reflect = self.reflect,
            max_iters = self.max_iters,
        )

        # get hydrogen indices from heavy atom indices
        self.hydrogen_map = self._get_hydrogen_map()

    def _get_hydrogen_map(self) -> List[List[int]]:

        """Compute map of each hydrogen to heavy atom
        """

        hydrogen_map = list()

        for atom in self.mol.GetAtoms():
            hydrogen_idcs = [at.GetIdx() for at in atom.GetNeighbors() if at.GetSymbol() == "H"]
            hydrogen_map.append(hydrogen_idcs)

        return hydrogen_map

    def _extend_match(
        self, heavy_match: List[int], transf: np.array, conf_idx_1: int, conf_idx_2: int
    ) -> List[int]:

        """Extend matching of heavy atoms to full mapping of all atoms

        heavy_match: List[int] Mapping heavy atom indices in conf_1 to corresponding atoms in conf_2
        transf: transformation matrix that optimally aligns conf_2 heavy atoms to conf_1

        Returns: full mapping of atom indices
        """

        full_match = [None for _ in range(self.mol.GetNumAtoms())]
        for no_h_idx_1, no_h_idx_2 in enumerate(heavy_match):
            idx_1 = self._match_h_removed[no_h_idx_1]
            idx_2 = self._match_h_removed[no_h_idx_2]

            full_match[idx_1] = idx_2

        conf_1 = self.mol.GetConformer(conf_idx_1)
        conf_2 = Chem.Conformer(self.mol.GetConformer(conf_idx_2))
        TransformConformer(conf_2, transf)

        coord_1: np.array = conf_1.GetPositions()
        coord_2: np.array = conf_2.GetPositions()

        for heavy_idx_1 in self._match_h_removed:
            heavy_idx_2 = full_match[heavy_idx_1]
            h_idcs_1 = self.hydrogen_map[heavy_idx_1]
            h_idcs_2 = self.hydrogen_map[heavy_idx_2]

            assert len(h_idcs_1) == len(h_idcs_2), 'Illegal Match Found'

            if len(h_idcs_1) == 0:
                continue
            if len(h_idcs_2) == 1:
                full_match[h_idcs_1[0]] = h_idcs_2[0]
                continue

            h_coords_1 = coord_1[h_idcs_1, :]
            min_permutation = None
            min_msd = np.inf

            for permuted_idcs in permutations(h_idcs_2):
                h_coords_2 = coord_2[permuted_idcs, :]
                msd = np.sum(np.square(h_coords_1 - h_coords_2))
                if msd<min_msd:
                    min_msd = msd
                    min_permutation = permuted_idcs

            for idx_1, idx_2 in zip(h_idcs_1, min_permutation):
                full_match[idx_1] = idx_2
        
        return full_match

    def _get_combinations(self):
        if self._combinations:
            return self._combinations

        conf_idcs = sorted([conf.GetId() for conf in self.mol.GetConformers()])

        for list_idx, (idx_1, idx_2) in enumerate(combinations(conf_idcs,2)):
            self._combinations.append((idx_1, idx_2))
            self._conf_idcs_to_list_idx[(idx_1, idx_2)] = list_idx

        return self._combinations

    def get_conf_idcs(self, list_idx: int) -> Tuple[int,int]:

        """Map index for the list returned by get_rmsd_matrix to pair of conformer indices
        """

        if not self._combinations:
            self._get_combinations()

        return self._combinations[list_idx]

    def get_list_idx(self, conf_idcs: Tuple[int,int]) -> int:

        """Maps pair of conformer indices to respective list index for the list return get_rmsd_matrix
        """

        if not self._combinations:
            self._get_combinations()

        return self._conf_idcs_to_list_idx[conf_idcs]

    def get_rmsd_matrix(self, num_cores: int = 1) -> List[float]:

        """Calculate matrix of RMSD between all conformers of a molecule
        """

        if not self._combinations:  
            self._get_combinations()

        if num_cores == 1:
            rmsd_matrix = [self._get_conf_rmsd(idcs) for idcs in self._combinations]
        else:
            with Pool(num_cores) as pool:
                rmsd_matrix = pool.map(self._get_conf_rmsd, self._combinations)

        return rmsd_matrix

    def _get_conf_rmsd(self, conf_idcs) -> float:
        return self.get_conformer_rmsd(conf_idcs)[0]

    def get_conformer_rmsd(self, conf_idcs: Tuple[int,int]) -> Tuple[float, np.array, List[int]]:

        """RMSD between two conformers
        Returns rmsd (float),  transf (np.array) transformation matrix conf_2 to conf_1
        match List[int] optimal match of atoms between two conformers
        """

        conf_1 = self.mol.GetConformer(conf_idcs[0])
        conf_2 = self.mol.GetConformer(conf_idcs[1])
        # find best heavy atom match
        _, transf, match = self.conf_rmsd.get_best_rmsd(*conf_idcs)

        full_match = self._extend_match(match, transf, *conf_idcs)

        # calculate final rmsd
        points_1 = [conf_1.GetAtomPosition(idx_1) for idx_1 in range(len(full_match))]
        points_2 = [conf_2.GetAtomPosition(idx_2) for idx_2 in full_match]
        ssr,transf = rdAlignment.GetAlignmentTransform(
            points_1, points_2, self.weights, self.reflect, self.max_iters
        )
        return np.sqrt(ssr / len(full_match)), transf, full_match 

#################################################################

def calc_hrmsd(mc,num_cores=1):
    h_conf_rmsd = HydrogenConformerRMSD(mc)
    rmsd_obj = h_conf_rmsd.get_rmsd_matrix(num_cores=num_cores)
    idx_map = h_conf_rmsd.get_conf_idcs

    # reshape
    num_confs = len(mc.GetConformers())
    rmsd_mat = np.zeros((num_confs, num_confs))
    for list_idx,rmsd in enumerate(rmsd_obj):
        idx_i, idx_j = idx_map(list_idx)
        rmsd_mat[idx_i, idx_j] = rmsd
    rmsd_mat += rmsd_mat.T
    return rmsd_mat

def find_duplicates(rmsd_array, threshold=0.3, exclude_diagonal=True):
    matches=np.argwhere(rmsd_array<threshold)
    if exclude_diagonal:
        matches = [m for m in matches if m[0]!=m[1]]
    return matches

def calculate_boltzmann_weights(energies):
    kT=298.15*0.0019872036 # kcal/mol
    energy_array = np.array(energies)
    weights = np.exp(-1.*energy_array/kT)
    return weights/np.sum(weights)

def update_mol_coords(mol, coords, conf_num=0):
    conf = mol.GetConformer(conf_num)
    for i in range(mol.GetNumAtoms()):
        x,y,z = coords[i,:]
        conf.SetAtomPosition(i,Point3D(x,y,z))

def get_rdkit_best_rmsd(idcs,mol):
    return Chem.rdMolAlign.GetBestRMS(mol,mol,refId=idcs[0], prbId=idcs[1])

def join_molecules(
    
    prb_mol: Chem.Mol,
    ref_mol: Chem.Mol,
    prb_cids: List[int] = None,
    ref_cids: List[int] = None,
) -> Tuple[Chem.Mol, Dict[int,int]]:

    """Join two rdkit mols of same structure with different conformers
    prb_mol and ref_mol are two rdkit mols
    prb_cids and ref_cids can be specific conformer idx to keep
    otherwise all are kept
    """

    if prb_cids is None:
        prb_cids = [conf.GetId() for conf in prb_mol.GetConformers()]

    mol_int = copy.deepcopy(ref_mol)

    all_ref_cids = [conf.GetId() for conf in ref_mol.GetConformers()]
    if ref_cids is None:
        ref_cids = all_ref_cids
    else:
        ref_cids_set = set(ref_cids)
        idcs_to_remove = set([idx for idx in all_ref_cids if idx not in ref_cids_set])
        remove_conformers(mol_int, idcs_to_remove)

    # now add the second mol as conformers
    cid_map = dict()
    atom_map = ref_mol.GetSubstructMatch(prb_mol)
    for prb_cid in prb_cids:
        prb_conformer = prb_mol.GetConformer(prb_cid)
        new_conformer = Chem.Conformer(mol_int.GetConformer(0))
        for prb_idx, ref_idx in enumerate(atom_map):
            position = prb_conformer.SetAtomPosition(prb_idx)
            new_conformer.SetAtomPosition(ref_idx, position)
        new_id = mol_int.AddConformer(new_conformer, assignId=True)
        cid_map[prb_cid] = new_id
    return mol_int, cid_map

def remove_conformers(mol: Chem.Mol, idcs_to_remove: Iterable[int]) -> None:

    for idx in idcs_to_remove:
        mol.RemoveConformer(idx)

    for new_idx, conformer in enumerate(mol.GetConformers()):
        conformer.SetId(new_idx)

def align_mol_to_reference(
    
    prb_mol: Chem.Mol,
    ref_mol: Chem.Mol,
    prb_cid: int = 0,
    ref_cid: int = 0,
) -> Tuple[Chem.Mol, float]:

    """Align mol to a reference, separating the hydrogens"""

    mol_name = prb_mol.GetProp('Name') if mol.HasProp('Name') else 'UNNAMED'
    mol_int, cid_map = join_molecules(prb_mol, ref_mol, prb_cids=[prb_cid], ref_cids=[ref_cid])
    
    h_conf_rmsd = HydrogenConformerRMSD(mol_int)
    new_id = cid_map[prb_cid]
    rmsd, transf, full_match = h_conf_rmsd.get_conformer_rmsd((0, new_id))

    remove_conformers(mol_int, set([0]))

    num_atoms = mol_int.GetNumAtoms()
    atom_positions = [mol_int.GetConformer(0).GetAtomPosition(atom_idx) for atom_idx in range(num_atoms)]
    prb_conformer = mol_int.GetConformer(0)
    for ref_idx, prb_idx in enumerate(full_match):
        prb_position = atom_positions[prb_idx]
        prb_conformer.SetAtomPosition(ref_idx, prb_position)
    TransformConformer(prb_conformer, transf)

    mol_int.SetProp('_Name',mol_name)
    return mol_int, rmsd

def fix_hydrogens(conformer, coordinates):

    """adapt hydrogen positions after geometry optimization
    Will fail if a different tautomer is input"""

    mol = conformer.GetOwningMol()
    new_conformer = Chem.Conformer(conformer)

    num_atoms = conformer.GetOwningMol().GetNumAtoms()
    num_heavy_atoms = conformer.GetOwningMol().GetNumHeavyAtoms()

    # container for hydrogens connected to given atom
    hs_from_parent = [queue.SimpleQueue() for _ in range(num_heavy_atoms)]
    # adjust heavy atoms
    for atom_idx in range(num_heavy_atoms):
        new_position = Geometry.Point3D(*coordinates[atom_idx, :])
        new_conformer.SetAtomPosition(atom_idx, new_position)
        atom = mol.GetAtomWithIdx(atom_idx)
        # store connect hydrogens for next stage
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == 'H':
                hs_from_parent[atom_idx].put(neighbor.GetIdx())

    # assume every hydrogen bonded to closet heavy atom
    # otherwise it must be a new tautomer so raise error
    dist_mat = get_distance_matrix(coordinates)
    for tm_h_atom_idx in range(num_heavy_atoms, num_atoms):
        # closest heavy atom
        parent_idx = np.argmin(dist_mat[tm_h_atom_idx, :num_heavy_atoms])
        # take one hydrogen bonded and adjust position
        h_queue = hs_from_parent[parent_idx]
        if h_queue.empty():
            _logger.error("tautomer shift?")
            raise RuntimeError("Geometry optimization introduced tautomer shift")
        h_idx = hs_from_parent[parent_idx].get()
        new_position = Geometry.Point3D(*coordinates[tm_h_atom_idx, :])
        new_conformer.SetAtomPosition(h_idx, new_position)

    return new_conformer

def get_distance_matrix(coords: np.array) -> np.array:

    assert coords.shape[1] == 3
    return distance_matrix(coords, coords)

