"""Открыть логи в TensorBoard."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import ROOT
from tensorboard import program

server = program.TensorBoard()
server.configure(argv=["tensorboard", "--logdir", str(ROOT / "logs"),
                       "--host", "localhost", "--port", "6006"] + sys.argv[1:])
server.main()
