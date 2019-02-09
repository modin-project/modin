set -x

cur_dir=$(pwd)

pushd .jenkins/1-generate-make-file
python3 shipyard.py modin-test.cfg.py > $cur_dir/Makefile
popd

make -j all
make_failed=$?

make publish_report

exit $make_failed


