#!/bin/bash



# NELL995

declare -a arr=(
worksfor 
organizationhiredperson 
organizationheadquarteredincity
athleteplayssport
teamplayssport
personborninlocation
athletehomestadium
organizationheadquarteredincity 
athleteplaysforteam
agentbelongstoorganization
teamplaysinleague
personleadsorganization)

# FB15K-237
#declare -a arr=(film@director@film
#film@film@language
#film@film@written_by
#location@capital_of_administrative_division@capital_of.@location@administrative_division_capital_relationship@administrative_division
#music@artist@origin
#organization@organization_founder@organizations_founded
#people@person@nationality
#people@person@place_of_birth
#sports@sports_team@sport
#tv@tv_program@languages)

# all kinship tasks
#declare -a arr=(term18 term8 term3 term1 term19 term16 term15 term7 term17 term12 term11 term20 term25 term13 term4 term2 term0 term5 term9 term21 term22 term14 term24 term6 term1)
#declare -a arr=(term19)
# all countries tasks
#declare -a arr=(countries_S1 countries_S2 countries_S3)

#file="freebase_log.txt"
file="nell_log_path4_3_combined.txt"
#file="kinship_log_hop3.txt"
#file="countries_log.txt"

if [ -f $file ] ; then
    rm $file
    touch $file
fi

for task_name in "${arr[@]}"
do
#    python main.py $task_name freebase $file
    python main.py $task_name nell $file



#    python main.py $task_name kinship $file
#    python main.py $task_name countries $file
done