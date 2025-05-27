import os
import random
import math
import logging
from multiprocessing import Pool
from rdkit import Chem
from rdkit.Chem import AllChem, RWMol, Draw
import subprocess
import pandas as pd
import tempfile

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mcts_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 用户配置参数
NUM_TREES = 10  # 独立 MCTS 树的数目，可在此修改
NUM_TOP_MOLECULES = 1000  # 输出前多少个高评分分子
INPUT_MCS_FILE = 'test_mcs.txt' # 骨架文件
INPUT_SUBS_FILE = 'subs_extract.txt' # 取代基文件
CHEMPROP_MODEL_PATH = 'your_trained_model/fold_0/model_0/best.pt'
CHEMPROP_PREDICT_SCRIPT = 'chemprop predict'
PREDICT_BATCH_SIZE = 100
MAX_SUBSTITUENTS = 4
MAX_CHILDREN_PER_EXPANSION = 8
TOTAL_ITERATIONS = 1000  # 每棵树的迭代次数
NUM_PROCESSES = 8
EXPLORATION_CONSTANT = 0.7
SCORE_THRESHOLD = 0.5
GENERATED_MOLECULES_FILE = 'your_generated_molecules.txt'
GENERATED_VISUALIZATION_DIR = 'your_generated_molecules'

# 设置工作目录
home_dir = os.path.expanduser('~')
working_dir = os.path.join(home_dir, 'Drug_design', 'monte')
generated_output_dir = os.path.join(working_dir, GENERATED_VISUALIZATION_DIR)
os.makedirs(working_dir, exist_ok=True)
os.makedirs(generated_output_dir, exist_ok=True)
os.chdir(working_dir)
logger.info(f"工作目录设置为：{os.getcwd()}")

# 加载SMILES或SMARTS
def load_smiles_from_file(file_path):
    smiles_list = []
    seen_smiles = set()
    try:
        with open(file_path, 'r') as f:
            for line in f:
                smiles = line.strip()
                if smiles:
                    if '*' in smiles:
                        mol = Chem.MolFromSmarts(smiles)
                        if mol:
                            smiles_list.append(smiles)
                        else:
                            logger.warning(f"无效 SMARTS：{smiles}")
                    else:
                        mol = Chem.MolFromSmiles(smiles, sanitize=False)
                        if mol:
                            try:
                                Chem.SanitizeMol(mol)
                                mol.UpdatePropertyCache(strict=False)
                                canon_smiles = Chem.MolToSmiles(mol, canonical=True)
                                if canon_smiles not in seen_smiles:
                                    seen_smiles.add(canon_smiles)
                                    smiles_list.append(canon_smiles)
                            except Exception as e:
                                logger.warning(f"无法规范化 SMILES {smiles}：{e}")
                        else:
                            logger.warning(f"无效 SMILES：{smiles}")
    except FileNotFoundError:
        logger.error(f"未找到文件 '{file_path}'")
        exit(1)
    except Exception as e:
        logger.error(f"读取 '{file_path}' 出错：{e}")
        exit(1)
    logger.info(f"从 {file_path} 加载了 {len(smiles_list)} 个唯一 SMILES/SMARTS")
    return smiles_list

# ChemProp 批量预测
def batch_predict(smiles_list, father_smiles_list):
    logger.info(f"开始批量预测 {len(smiles_list)} 个分子")
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_input:
        df_input = pd.DataFrame({'SMILES': smiles_list, 'Father_SMILES': father_smiles_list})
        df_input.to_csv(tmp_input.name, index=False)
        input_path = tmp_input.name
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_output:
        output_path = tmp_output.name
    cmd = f"{CHEMPROP_PREDICT_SCRIPT} --test-path {input_path} --model-path {CHEMPROP_MODEL_PATH} --preds-path {output_path}"
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"ChemProp 预测失败：{e}")
        return [0.0] * len(smiles_list)
    try:
        df_output = pd.read_csv(output_path)
        scores = df_output['pred_0'].tolist()
        return scores
    except Exception as e:
        logger.error(f"读取预测结果出错：{e}")
        return [0.0] * len(smiles_list)
    finally:
        os.remove(input_path)
        os.remove(output_path)

# 添加显式氢原子
def add_explicit_hydrogens(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            logger.error(f"无法解析 SMILES: {smiles}")
            return None
        Chem.SanitizeMol(mol)
        mol.UpdatePropertyCache(strict=False)
        Chem.Kekulize(mol, clearAromaticFlags=True)
        mol = Chem.AddHs(mol, explicitOnly=False)
        return Chem.MolToSmiles(mol, canonical=True, allHsExplicit=True)
    except Exception as e:
        logger.error(f"处理 SMILES {smiles} 时出错: {e}")
        return None

# 识别结构单元（环和非环链）
def identify_structural_units(mol):
    if not mol:
        logger.warning("无法识别结构单元：分子无效")
        return []
    
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    structural_units = [set(ring) for ring in atom_rings]
    
    all_atoms = set(range(mol.GetNumAtoms()))
    ring_atoms = set.union(*structural_units) if structural_units else set()
    non_ring_atoms = all_atoms - ring_atoms
    
    if non_ring_atoms:
        structural_units.append(non_ring_atoms)
    
    return structural_units

# MCTS 节点类
class MCTSNode:
    def __init__(self, smiles, mcs_atom_indices, father_smiles, structural_units=None, parent=None, action=None, substituent_count=0, unit_substituent_counts=None, substituent_atoms=None):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if self.mol:
            try:
                Chem.SanitizeMol(self.mol)
                Chem.Kekulize(self.mol, clearAromaticFlags=True)
                self.mol.UpdatePropertyCache(strict=False)
            except Exception as e:
                logger.warning(f"无法规范化节点 SMILES {smiles}：{e}")
                self.mol = None
        self.mcs_atom_indices = mcs_atom_indices
        self.father_smiles = father_smiles
        self.structural_units = structural_units if structural_units is not None else identify_structural_units(self.mol)
        self.unit_substituent_counts = unit_substituent_counts if unit_substituent_counts is not None else {i: 0 for i in range(len(self.structural_units))}
        self.substituent_atoms = substituent_atoms if substituent_atoms is not None else set()
        self.parent = parent
        self.children = []
        self.action = action
        self.visits = 0
        self.total_reward = 0.0
        self.substituent_count = substituent_count
        self.available_actions = self.get_available_actions()

    def get_available_actions(self):
        if not self.mol:
            logger.warning(f"节点无有效分子：SMILES={self.smiles}")
            return []
        if self.substituent_count >= MAX_SUBSTITUENTS:
            return []
        
        actions = []
        mol = Chem.MolFromSmiles(self.smiles, sanitize=False)
        Chem.SanitizeMol(mol)
        Chem.Kekulize(mol, clearAromaticFlags=True)
        mol.UpdatePropertyCache(strict=False)
        mol = Chem.AddHs(mol, explicitOnly=False)
        
        min_count = min(self.unit_substituent_counts.values())
        candidate_unit_indices = [i for i, count in self.unit_substituent_counts.items() if count == min_count]
        
        for unit_idx in candidate_unit_indices:
            unit_atoms = self.structural_units[unit_idx]
            for atom in mol.GetAtoms():
                idx = atom.GetIdx()
                if idx not in unit_atoms or idx in self.substituent_atoms:
                    continue
                symbol = atom.GetSymbol()
                num_h = atom.GetTotalNumHs()
                explicit_h = sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H')
                in_mcs = idx in self.mcs_atom_indices
                if num_h > 0 or explicit_h > 0:
                    for sub_smiles in SUBSTITUENTS:
                        actions.append((idx, sub_smiles, unit_idx))
        
        return actions

    def is_terminal(self):
        return self.substituent_count >= MAX_SUBSTITUENTS or len(self.available_actions) == 0

    def ucb1(self):
        if self.visits == 0:
            return float('inf')
        parent_visits = self.parent.visits if self.parent else 1
        return (self.total_reward / self.visits) + EXPLORATION_CONSTANT * math.sqrt(math.log(parent_visits) / self.visits)

# MCTS 算法类
class MCTS:
    def __init__(self, mcs_smiles, substituents, iterations=263):
        self.mcs_smiles = add_explicit_hydrogens(mcs_smiles) or mcs_smiles
        self.substituents = substituents
        self.iterations = iterations
        logger.info(f"初始化 MCTS：MCS={self.mcs_smiles}, iterations={iterations}")

    def search(self):
        mol = Chem.MolFromSmiles(self.mcs_smiles, sanitize=False)
        if not mol:
            logger.error(f"无法解析 MCS：{self.mcs_smiles}")
            return []
        try:
            Chem.SanitizeMol(mol)
            mol.UpdatePropertyCache(strict=False)
        except Exception as e:
            logger.error(f"无法规范化 MCS {self.mcs_smiles}：{e}")
            return []
        mcs_atom_indices = set(range(mol.GetNumAtoms()))
        structural_units = identify_structural_units(mol)
        root = MCTSNode(
            smiles=self.mcs_smiles,
            mcs_atom_indices=mcs_atom_indices,
            father_smiles=self.mcs_smiles,
            structural_units=structural_units,
            substituent_count=0,
            substituent_atoms=set()
        )
        best_molecules = []
        batch_nodes = []
        logger.info(f"开始 MCTS 搜索，根节点 SMILES={self.mcs_smiles}, 结构单元数={len(structural_units)}")

        for iteration in range(self.iterations):
            node = self.select(root)
            new_nodes = self.expand(node)
            batch_nodes.extend(new_nodes)
            if len(batch_nodes) >= PREDICT_BATCH_SIZE:
                scores = self.simulate_batch(batch_nodes)
                for new_node, score in zip(batch_nodes, scores):
                    self.backpropagate(new_node, score)
                    if new_node.smiles and score > SCORE_THRESHOLD:
                        best_molecules.append((new_node.smiles, score))
                batch_nodes = []
        if batch_nodes:
            scores = self.simulate_batch(batch_nodes)
            for new_node, score in zip(batch_nodes, scores):
                self.backpropagate(new_node, score)
                if new_node.smiles and score > SCORE_THRESHOLD:
                    best_molecules.append((new_node.smiles, score))
        logger.info(f"搜索完成，生成 {len(best_molecules)} 个分子")
        return best_molecules

    def select(self, node):
        while not node.is_terminal() and node.children:
            node = max(node.children, key=lambda c: c.ucb1())
        return node

    def expand(self, node):
        if not node.available_actions:
            return [node]
        num_children = min(len(node.available_actions), MAX_CHILDREN_PER_EXPANSION)
        actions = random.sample(node.available_actions, num_children)
        new_nodes = []
        for action in actions:
            attachment_idx, sub_smiles, unit_idx = action
            new_smiles = self.attach_substituent(node.smiles, attachment_idx, sub_smiles, unit_idx, node.unit_substituent_counts, node.substituent_atoms.copy())
            if not new_smiles:
                continue
            new_mcs_atom_indices = node.mcs_atom_indices.copy()
            new_unit_counts = node.unit_substituent_counts.copy()
            new_unit_counts[unit_idx] += 1
            mol = Chem.MolFromSmiles(new_smiles)
            new_atom_idx = mol.GetNumAtoms() - 1
            new_substituent_atoms = node.substituent_atoms.copy()
            new_substituent_atoms.add(new_atom_idx)
            child = MCTSNode(
                smiles=new_smiles,
                mcs_atom_indices=new_mcs_atom_indices,
                father_smiles=node.father_smiles,
                structural_units=node.structural_units,
                parent=node,
                action=action,
                substituent_count=node.substituent_count + 1,
                unit_substituent_counts=new_unit_counts,
                substituent_atoms=new_substituent_atoms
            )
            node.available_actions.remove(action)
            node.children.append(child)
            new_nodes.append(child)
        return new_nodes if new_nodes else [node]

    def simulate_batch(self, nodes):
        smiles_list = [node.smiles for node in nodes]
        father_smiles_list = [node.father_smiles for node in nodes]
        return batch_predict(smiles_list, father_smiles_list)

    def backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def attach_substituent(self, mol_smiles, attachment_idx, sub_smiles, unit_idx, unit_substituent_counts, substituent_atoms):
        mol = Chem.MolFromSmiles(mol_smiles, sanitize=False)
        sub_mol = Chem.MolFromSmarts(sub_smiles)
        if not mol or not sub_mol:
            return None
        try:
            Chem.SanitizeMol(mol)
            Chem.SanitizeMol(sub_mol)
            mol.UpdatePropertyCache(strict=False)
            sub_mol.UpdatePropertyCache(strict=False)
        except Exception as e:
            return None

        star_idx = None
        for atom in sub_mol.GetAtoms():
            if atom.GetSymbol() == '*':
                star_idx = atom.GetIdx()
                break
        if star_idx is None:
            return None

        rw_mol = RWMol(Chem.AddHs(mol))
        h_idx = None
        target_atom = None
        for atom in rw_mol.GetAtoms():
            if atom.GetIdx() == attachment_idx:
                target_atom = atom
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetSymbol() == 'H':
                        h_idx = neighbor.GetIdx()
                        break
                break
        if h_idx is None or target_atom is None:
            return None

        rw_mol.RemoveAtom(h_idx)
        star_neighbors = [n for n in sub_mol.GetAtomWithIdx(star_idx).GetNeighbors()]
        if len(star_neighbors) == 0:
            return None

        connect_atom = star_neighbors[0]
        connect_symbol = connect_atom.GetSymbol()

        if sub_mol.GetNumAtoms() == 2:
            if connect_symbol in ['F', 'Cl', 'Br', 'I']:
                new_atom = Chem.Atom(connect_atom.GetAtomicNum())
                new_idx = rw_mol.AddAtom(new_atom)
                rw_mol.AddBond(attachment_idx, new_idx, Chem.BondType.SINGLE)
            elif connect_symbol == 'O':
                new_atom = Chem.Atom(8)
                new_atom.SetNumExplicitHs(1)
                new_idx = rw_mol.AddAtom(new_atom)
                rw_mol.AddBond(attachment_idx, new_idx, Chem.BondType.SINGLE)
            elif connect_symbol == 'C':
                new_atom = Chem.Atom(6)
                new_atom.SetNumExplicitHs(3)
                new_idx = rw_mol.AddAtom(new_atom)
                rw_mol.AddBond(attachment_idx, new_idx, Chem.BondType.SINGLE)
            else:
                return None
        else:
            sub_mol_without_star = RWMol(sub_mol)
            sub_mol_without_star.RemoveAtom(star_idx)
            sub_atoms = sub_mol_without_star.GetNumAtoms()
            if sub_atoms == 0:
                return None
            new_atom_idx = rw_mol.GetNumAtoms()
            sub_atom_map = {}
            for atom in sub_mol_without_star.GetAtoms():
                new_idx = rw_mol.AddAtom(atom)
                sub_atom_map[atom.GetIdx()] = new_idx
            for bond in sub_mol_without_star.GetBonds():
                rw_mol.AddBond(
                    sub_atom_map[bond.GetBeginAtomIdx()],
                    sub_atom_map[bond.GetEndAtomIdx()],
                    bond.GetBondType()
                )
            connect_atom_idx = sub_atom_map[connect_atom.GetIdx()]
            rw_mol.AddBond(attachment_idx, connect_atom_idx, Chem.BondType.SINGLE)

        try:
            Chem.SanitizeMol(rw_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            Chem.Kekulize(rw_mol, clearAromaticFlags=True)
            return Chem.MolToSmiles(rw_mol, canonical=True)
        except Exception as e:
            return None

# 并行 MCTS 运行函数
def run_mcts(args):
    mcs_smiles, mcs_index, tree_id = args
    logger.info(f"开始 MCTS 进程：MCS编号={mcs_index}, 树编号={tree_id}")
    mcts = MCTS(mcs_smiles=mcs_smiles, substituents=SUBSTITUENTS, iterations=TOTAL_ITERATIONS)
    try:
        results = mcts.search()
    except Exception as e:
        logger.error(f"MCTS 进程失败：MCS编号={mcs_index}, 树编号={tree_id}, 错误={e}")
        return []
    results_with_metadata = [(smiles, score, mcs_index, tree_id) for smiles, score in results]
    logger.info(f"MCTS 进程完成：MCS编号={mcs_index}, 树编号={tree_id}, 生成 {len(results)} 个分子")
    return results_with_metadata

# 保存和可视化分子
def save_and_visualize_molecules(molecules):
    logger.info(f"保存和可视化 {len(molecules)} 个分子")
    with open(GENERATED_MOLECULES_FILE, 'w') as f:
        for i, (smiles, (score, mcs_index, tree_id)) in enumerate(molecules):
            f.write(f"{smiles}\t{score}\t{mcs_index}\t{tree_id}\n")
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    img_path = os.path.join(generated_output_dir, f"molecule_{i+1}_score_{score:.3f}_mcs_{mcs_index}_tree_{tree_id}.png")
                    Draw.MolToFile(mol, img_path, size=(300, 300))
                else:
                    logger.warning(f"无法生成图像：无效 SMILES={smiles}")
            except Exception as e:
                logger.error(f"生成分子图像失败：SMILES={smiles}, 错误={e}")

# 主程序
if __name__ == "__main__":
    MCS_LIST = load_smiles_from_file(INPUT_MCS_FILE)
    SUBSTITUENTS = load_smiles_from_file(INPUT_SUBS_FILE)
    logger.info(f"加载了 {len(MCS_LIST)} 个 MCS 和 {len(SUBSTITUENTS)} 个取代基")
    
    tasks = [(mcs_smiles, mcs_index + 1, tree_id) for mcs_index, mcs_smiles in enumerate(MCS_LIST) for tree_id in range(1, NUM_TREES + 1)]
    logger.info(f"创建了 {len(tasks)} 个 MCTS 任务")
    
    with Pool(NUM_PROCESSES) as p:
        all_results = p.map(run_mcts, tasks)
    
    all_molecules = [item for sublist in all_results for item in sublist]
    logger.info(f"从所有树中收集了 {len(all_molecules)} 个分子")
    
    molecule_data = {}
    for smiles, score, mcs_index, tree_id in all_molecules:
        if smiles not in molecule_data:
            molecule_data[smiles] = (score, mcs_index, tree_id)
    
    sorted_molecules = sorted(molecule_data.items(), key=lambda x: x[1][0], reverse=True)[:NUM_TOP_MOLECULES]
    logger.info(f"合并后得到 {len(sorted_molecules)} 个唯一分子")
    
    save_and_visualize_molecules(sorted_molecules)
    
    logger.info(f"MCTS 完成，结果保存至 {GENERATED_MOLECULES_FILE}，图像保存至 {generated_output_dir}")
