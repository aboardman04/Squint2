import envs
import sys
import runpy

sys.argv = ['mani_skill.examples.demo_random_action', '-e', 'SeparateInstruments-v0', '--render-mode', 'human']
runpy.run_module('mani_skill.examples.demo_random_action', run_name='__main__')
