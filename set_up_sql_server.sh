set -e

sudo docker pull mcr.microsoft.com/mssql/server:2019-latest
echo "mvashishtha pulled docker image"
sudo docker run -d --name example_sql_server -e 'ACCEPT_EULA=Y' -e 'SA_PASSWORD=Strong.Pwd-123' -p 1433:1433 mcr.microsoft.com/mssql/server:2019-latest
echo "mvashishtha started server"
sudo docker cp test_1000x256.csv $(sudo docker container ls -f name=example_sql_server -q):/test_1000x256.csv
echo "mvashishtha copied file"
# https://citizix.com/how-to-install-ms-sql-server-2019-on-ubuntu-20-04/
curl https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
echo "mvashishtha first curl"
curl https://packages.microsoft.com/config/ubuntu/20.04/prod.list | sudo tee /etc/apt/sources.list.d/msprod.list
echo "mvashishtha second curl"
sudo apt-cache update
echo "mvashishtha updated"
sudo apt-cache install mssql-tools unixodbc-dev -y
echo "mvashishtha got tools"
sudo apt-cache install net-tools
echo "got net-tools"
sudo netstat -tulpn | grep :1433
echo "ran netstat"
sudo docker container ls
echo "listed docker containers"
/opt/mssql-tools/bin/sqlcmd -S 0.0.0.0 -U sa -P Strong.Pwd-123 -Q "CREATE TABLE test_1000x256 (col0 int, col1 int, col2 int, col3 int, col4 int, col5 int, col6 int, col7 int, col8 int, col9 int, col10 int, col11 int, col12 int, col13 int, col14 int, col15 int, col16 int, col17 int, col18 int, col19 int, col20 int, col21 int, col22 int, col23 int, col24 int, col25 int, col26 int, col27 int, col28 int, col29 int, col30 int, col31 int, col32 int, col33 int, col34 int, col35 int, col36 int, col37 int, col38 int, col39 int, col40 int, col41 int, col42 int, col43 int, col44 int, col45 int, col46 int, col47 int, col48 int, col49 int, col50 int, col51 int, col52 int, col53 int, col54 int, col55 int, col56 int, col57 int, col58 int, col59 int, col60 int, col61 int, col62 int, col63 int, col64 int, col65 int, col66 int, col67 int, col68 int, col69 int, col70 int, col71 int, col72 int, col73 int, col74 int, col75 int, col76 int, col77 int, col78 int, col79 int, col80 int, col81 int, col82 int, col83 int, col84 int, col85 int, col86 int, col87 int, col88 int, col89 int, col90 int, col91 int, col92 int, col93 int, col94 int, col95 int, col96 int, col97 int, col98 int, col99 int, col100 int, col101 int, col102 int, col103 int, col104 int, col105 int, col106 int, col107 int, col108 int, col109 int, col110 int, col111 int, col112 int, col113 int, col114 int, col115 int, col116 int, col117 int, col118 int, col119 int, col120 int, col121 int, col122 int, col123 int, col124 int, col125 int, col126 int, col127 int, col128 int, col129 int, col130 int, col131 int, col132 int, col133 int, col134 int, col135 int, col136 int, col137 int, col138 int, col139 int, col140 int, col141 int, col142 int, col143 int, col144 int, col145 int, col146 int, col147 int, col148 int, col149 int, col150 int, col151 int, col152 int, col153 int, col154 int, col155 int, col156 int, col157 int, col158 int, col159 int, col160 int, col161 int, col162 int, col163 int, col164 int, col165 int, col166 int, col167 int, col168 int, col169 int, col170 int, col171 int, col172 int, col173 int, col174 int, col175 int, col176 int, col177 int, col178 int, col179 int, col180 int, col181 int, col182 int, col183 int, col184 int, col185 int, col186 int, col187 int, col188 int, col189 int, col190 int, col191 int, col192 int, col193 int, col194 int, col195 int, col196 int, col197 int, col198 int, col199 int, col200 int, col201 int, col202 int, col203 int, col204 int, col205 int, col206 int, col207 int, col208 int, col209 int, col210 int, col211 int, col212 int, col213 int, col214 int, col215 int, col216 int, col217 int, col218 int, col219 int, col220 int, col221 int, col222 int, col223 int, col224 int, col225 int, col226 int, col227 int, col228 int, col229 int, col230 int, col231 int, col232 int, col233 int, col234 int, col235 int, col236 int, col237 int, col238 int, col239 int, col240 int, col241 int, col242 int, col243 int, col244 int, col245 int, col246 int, col247 int, col248 int, col249 int, col250 int, col251 int, col252 int, col253 int, col254 int, col255 int);"
echo "mvashishtha ran first command"
/opt/mssql-tools/bin/sqlcmd -S 0.0.0.0 -U sa -P Strong.Pwd-123 -Q "BULK INSERT test_1000x256 FROM '/test_1000x256.csv' WITH (firstrow = 2, fieldterminator = ',')"
echo "mvashishtha ran second command."