import os
from pyscons import PYTOOL

# Faster dependency checking -- important for big hdf5 files
Decider('timestamp-match')

####################### BUILDERS ###################################

# This is how to make plots from correspondingly-named python scripts
plot_figures = Builder(
    action='pipenv run python ${SOURCES[1:]} $TARGET',
    emitter=lambda target, source, env: \
    (target, ['matplotlibrc', 'scripts/plot_${TARGET.file}.py'] + source))

#####################################################################

# Set up environment with custom builder and normal path (needed to
# find right python to use). PYTOOL provides implicit dependency
# tracking of Python modules.
env = Environment(tools=['default', PYTOOL()],
                  ENV=os.environ,
                  BUILDERS={'Figure': plot_figures})

# Add current directory to path so plotting scripts are correctly
# tracked as dependencies
env['ENV']['PATH'] = '{}:{}'.format(Dir('.').srcnode().abspath,
                                    env['ENV']['PATH'])

# Enable SyncTeX
env.AppendUnique(PDFLATEXFLAGS='-synctex=1')

######################################################################
## INTRODUCTION ######################################################
######################################################################

env.Figure('paper/figures/thrust_lin_example.pdf', [
    'data/NREL5MW_simpblade.yaml',
    'data/oc3_aerofoils.npz',
])

######################################################################
## WAKE DYNAMICS #####################################################
######################################################################

env.Figure('paper/figures/wake_harmonic_solutions.pdf', [
    'data/wake/inflow_derivatives.npz',
])

env.Figure('paper/figures/wake_contours.pdf', [
    'data/wake/inflow_derivatives.npz',
])

######################################################################
## AERODYNAMICS & STRUCTURAL DYNAMICS ################################
######################################################################

env.Figure('paper/figures/linearisation_loop_examples.pdf', [
    'data/aero/bladed_harmonic_wind.h5'
])

env.Figure('paper/figures/linearisation_sine_examples.pdf', [
    'data/aero/bladed_harmonic_wind.h5'
])

# Actually produces linearisation_errors_0.pdf and linearisation_errors_1.pdf
env.Figure('paper/figures/linearisation_errors.pdf', [
    'data/aero/bladed_harmonic_wind.h5'
])

env.Figure('paper/figures/linearisation_loops_along_blade.pdf', [
    'data/aero/bladed_harmonic_wind.h5',
    'data/NREL5MW_simpblade_model.yaml',
])

######################################################################
## CONTROL SYSTEM ####################################################
######################################################################

env.Figure('paper/figures/harmonic_torque_examples.pdf', [
    'data/NREL5MW_simpblade_model.yaml'
])

env.Figure('paper/figures/torque_nonlinearity.pdf', [
    'data/NREL5MW_simpblade_model.yaml'
])

env.Figure('paper/figures/control_linearisation_errors.pdf', [
    'data/control/simpblade_frozen2.h5',
    'data/control/simpblade_frozen_harmonic2_noforceconstpower.h5',
])

# Actually produces control_linearisation_examples_{lowfreq,midfreq,highfreq}
env.Figure('paper/figures/control_linearisation_examples.pdf', [
    'data/control/simpblade_frozen2.h5',
    'data/control/simpblade_frozen_harmonic2_noforceconstpower.h5',
])

######################################################################

pdf = env.PDF('paper/paper.tex')
Depends(pdf, Glob('paper/figures/*.pdf'))
Default(pdf)
