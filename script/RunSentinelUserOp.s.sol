// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Script.sol";
import "forge-std/console2.sol";

import {IEntryPoint} from
    "account-abstraction/interfaces/IEntryPoint.sol";
import {PackedUserOperation} from
    "account-abstraction/interfaces/PackedUserOperation.sol";

import {SentinelAccountFactory} from "../src/SentinelAccountFactory.sol";
import {SentinelTarget} from "../src/SentinelTarget.sol";

contract RunSentinelUserOp is Script {
    function run() external {
        uint256 privateKey = vm.envUint("PRIVATE_KEY");

        IEntryPoint ep =
            IEntryPoint(vm.envAddress("ENTRY_POINT"));

        SentinelAccountFactory factory =
            SentinelAccountFactory(vm.envAddress("SENTINEL_FACTORY"));

        SentinelTarget target =
            SentinelTarget(vm.envAddress("TARGET"));

        address owner = vm.addr(privateKey);

        // Use a fresh salt for this attempt.
        uint256 salt = 779;

        address sender = factory.getAddress(owner, salt);

        console2.log("Owner:", owner);
        console2.log("Counterfactual account:", sender);
        console2.log("Target:", address(target));

        require(target.value() == 0, "Target value is not zero");

        vm.startBroadcast(privateKey);

        // Fund the counterfactual account's EntryPoint deposit.
        ep.depositTo{value: 1 ether}(sender);

        PackedUserOperation[] memory ops =
            new PackedUserOperation[](1);

        PackedUserOperation memory op;

        op.sender = sender;
        op.nonce = ep.getNonce(sender, 0);

        op.initCode = abi.encodePacked(
            address(factory),
            abi.encodeCall(
                SentinelAccountFactory.createAccount,
                (owner, salt)
            )
        );

        bytes memory targetCall = abi.encodeCall(
            SentinelTarget.setValue,
            (42)
        );

        op.callData = abi.encodeWithSignature(
            "execute(address,uint256,bytes)",
            address(target),
            0,
            targetCall
        );

        // PackedUserOperation gas limits:
        // high 128 bits = verification gas
        // low 128 bits  = call gas
        op.accountGasLimits = bytes32(
            (uint256(300_000) << 128) | uint256(200_000)
        );

        op.preVerificationGas = 50_000;

        // high 128 bits = max priority fee
        // low 128 bits  = max fee
        op.gasFees = bytes32(
            (uint256(1 gwei) << 128) | uint256(2 gwei)
        );

        op.paymasterAndData = "";

        bytes32 userOpHash = ep.getUserOpHash(op);

        (uint8 v, bytes32 r, bytes32 s) =
            vm.sign(privateKey, userOpHash);

        op.signature = abi.encodePacked(r, s, v);

        ops[0] = op;

        console2.log("UserOperation hash:");
        console2.logBytes32(userOpHash);

        ep.handleOps(ops, payable(owner));

        vm.stopBroadcast();

        console2.log("Account code length:", sender.code.length);
        console2.log("Final target value:", target.value());

        require(sender.code.length > 0, "Account not deployed");
        require(target.value() == 42, "Target update failed");

        console2.log("SUCCESS: ERC-4337 UserOperation executed.");
    }
}
