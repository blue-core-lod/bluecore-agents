import pathlib
import sys

root_directory = pathlib.Path(__file__).parent.parent
src_dir = root_directory / "src/"
sys.path.append(str(src_dir))
